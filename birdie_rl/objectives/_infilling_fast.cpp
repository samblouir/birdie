#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <exception>
#include <random>
#include <thread>
#include <vector>

namespace {

static int seq_to_i32_vector(PyObject* seq_obj, std::vector<int32_t>& out) {
	PyObject* seq = PySequence_Fast(seq_obj, "expected a sequence of ints");
	if (!seq) {
		return 0;
	}
	Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
	out.clear();
	out.reserve(static_cast<size_t>(n));
	PyObject** items = PySequence_Fast_ITEMS(seq);
	for (Py_ssize_t i = 0; i < n; i++) {
		long v = PyLong_AsLong(items[i]);
		if (PyErr_Occurred()) {
			Py_DECREF(seq);
			return 0;
		}
		out.push_back(static_cast<int32_t>(v));
	}
	Py_DECREF(seq);
	return 1;
}

static PyObject* i32_vector_to_bytes(const std::vector<int32_t>& v) {
	const Py_ssize_t nbytes = static_cast<Py_ssize_t>(v.size() * sizeof(int32_t));
	PyObject* bytes_obj = PyBytes_FromStringAndSize(nullptr, nbytes);
	if (!bytes_obj) {
		return nullptr;
	}
	char* buf = PyBytes_AS_STRING(bytes_obj);
	std::memcpy(buf, v.data(), static_cast<size_t>(nbytes));
	return bytes_obj;
}

struct ScopedGILRelease {
	PyThreadState* state;
	ScopedGILRelease() : state(PyEval_SaveThread()) {}
	~ScopedGILRelease() { PyEval_RestoreThread(state); }
};

static inline std::uint64_t splitmix64(std::uint64_t x) {
	x += 0x9e3779b97f4a7c15ULL;
	x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
	x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
	return x ^ (x >> 31);
}

static inline std::uint64_t derive_candidate_seed(std::uint64_t base_seed, int candidate_idx) {
	if (candidate_idx <= 0) {
		return base_seed;
	}
	return splitmix64(base_seed + static_cast<std::uint64_t>(candidate_idx));
}

struct CandidateResult {
	bool ok = false;
	bool used_fallback = false;
	int masked_count = 0;
	int original_length = 0;
	std::vector<int32_t> input;
	std::vector<int32_t> label;
	std::vector<int32_t> unused;
};

static CandidateResult run_infilling_candidate(
	const std::vector<int32_t>& encoded_input,
	int remaining_space,
	std::uint64_t rng_seed,
	double min_corruption_rate,
	double max_corruption_rate,
	double mean_tokens_per_span,
	double min_mean_tokens_per_span,
	double max_mean_tokens_per_span,
	int max_mask_spans,
	int max_attempts,
	const std::vector<int32_t>& prompt_toks,
	const std::vector<int32_t>& paradigm_end_toks,
	const std::vector<std::vector<int32_t>>& placeholder_bank,
	int max_n_tokens_to_process,
	const std::atomic<bool>* cancel) {
	CandidateResult out;
	out.ok = true;

	const int n_tokens = static_cast<int>(encoded_input.size());
	const int prompt_len = static_cast<int>(prompt_toks.size());
	const int paradigm_end_len = static_cast<int>(paradigm_end_toks.size());

	std::mt19937_64 rng(rng_seed);
	std::uniform_real_distribution<double> uni01(0.0, 1.0);
	std::uniform_real_distribution<double> uni_cor(min_corruption_rate, max_corruption_rate);
	std::poisson_distribution<int> poisson(mean_tokens_per_span);

	auto should_cancel = [&]() -> bool {
		return cancel && cancel->load(std::memory_order_relaxed);
	};

	auto sample_span_length = [&](int current_text_idx) -> int {
		const int limit = max_n_tokens_to_process - current_text_idx;
		double raw = static_cast<double>(poisson(rng));
		raw = std::max(raw, min_mean_tokens_per_span);
		raw = std::min(raw, max_mean_tokens_per_span);
		raw = std::min(raw, static_cast<double>(std::max(0, limit)));
		return static_cast<int>(std::max(1.0, raw));
	};

	for (int attempt = 0; attempt < max_attempts; attempt++) {
		if (should_cancel()) {
			out.ok = false;
			return out;
		}
		out.input.clear();
		out.label.clear();
		out.unused.clear();
		out.input.reserve(static_cast<size_t>(prompt_len + max_n_tokens_to_process));
		out.input.insert(out.input.end(), prompt_toks.begin(), prompt_toks.end());

		int placeholders_inserted = 0;
		int masked_count = 0;
		int text_idx = 0;

		while (text_idx < max_n_tokens_to_process && placeholders_inserted < max_mask_spans) {
			if (should_cancel()) {
				out.ok = false;
				return out;
			}
			const int input_len = static_cast<int>(out.input.size());
			const int label_len_total = static_cast<int>(out.label.size()) + paradigm_end_len;
			if (input_len + label_len_total >= remaining_space) {
				break;
			}

			const double local_cor = uni_cor(rng);
			const int span_len = sample_span_length(text_idx);
			const double prob_to_mask = local_cor / static_cast<double>(span_len);

			const bool can_mask = (max_n_tokens_to_process - text_idx) >= span_len;
			const bool do_mask = can_mask && (uni01(rng) < prob_to_mask);

			if (do_mask) {
				const std::vector<int32_t>& ph = placeholder_bank[static_cast<size_t>(placeholders_inserted)];
				const int ph_len = static_cast<int>(ph.size());
				const int prospective_input_len = input_len + ph_len;
				const int prospective_label_len_total =
					static_cast<int>(out.label.size()) + ph_len + span_len + paradigm_end_len;
				if (prospective_input_len + prospective_label_len_total <= remaining_space) {
					out.input.insert(out.input.end(), ph.begin(), ph.end());
					out.label.insert(out.label.end(), ph.begin(), ph.end());
					out.label.insert(
						out.label.end(),
						encoded_input.begin() + text_idx,
						encoded_input.begin() + (text_idx + span_len));
					text_idx += span_len;
					placeholders_inserted += 1;
					masked_count += span_len;
					continue;
				}
			}

			if (input_len + 1 + label_len_total <= remaining_space) {
				out.input.push_back(encoded_input[static_cast<size_t>(text_idx)]);
				text_idx += 1;
			} else {
				break;
			}
		}

		if (placeholders_inserted > 0) {
			if (paradigm_end_len > 0) {
				if (static_cast<int>(out.input.size()) + static_cast<int>(out.label.size()) + paradigm_end_len <= remaining_space) {
					out.label.insert(out.label.end(), paradigm_end_toks.begin(), paradigm_end_toks.end());
				}
			}
			if (text_idx < n_tokens) {
				out.unused.insert(out.unused.end(), encoded_input.begin() + text_idx, encoded_input.end());
			}
			out.used_fallback = false;
			out.masked_count = masked_count;
			out.original_length = text_idx;
			return out;
		}
	}

	// Forced-mask fallback (guarantees non-empty label_ids).
	const std::vector<int32_t>& ph0 = placeholder_bank[0];
	const int ph0_len = static_cast<int>(ph0.size());
	const int budget_for_text = remaining_space - prompt_len - (2 * ph0_len) - paradigm_end_len;
	if (budget_for_text <= 0) {
		out.ok = false;
		return out;
	}
	int total_text_tokens_to_use = std::min(max_n_tokens_to_process, budget_for_text);
	total_text_tokens_to_use = std::max(1, total_text_tokens_to_use);

	int mask_len = sample_span_length(0);
	mask_len = std::max(1, std::min(mask_len, total_text_tokens_to_use));

	out.input.clear();
	out.label.clear();
	out.unused.clear();
	out.input.reserve(static_cast<size_t>(prompt_len + ph0_len + total_text_tokens_to_use));
	out.input.insert(out.input.end(), prompt_toks.begin(), prompt_toks.end());
	out.input.insert(out.input.end(), ph0.begin(), ph0.end());
	out.input.insert(
		out.input.end(),
		encoded_input.begin() + mask_len,
		encoded_input.begin() + total_text_tokens_to_use);

	out.label.insert(out.label.end(), ph0.begin(), ph0.end());
	out.label.insert(out.label.end(), encoded_input.begin(), encoded_input.begin() + mask_len);
	if (paradigm_end_len > 0) {
		if (static_cast<int>(out.input.size()) + static_cast<int>(out.label.size()) + paradigm_end_len <= remaining_space) {
			out.label.insert(out.label.end(), paradigm_end_toks.begin(), paradigm_end_toks.end());
		}
	}
	if (total_text_tokens_to_use < n_tokens) {
		out.unused.insert(out.unused.end(), encoded_input.begin() + total_text_tokens_to_use, encoded_input.end());
	}

	out.used_fallback = true;
	out.masked_count = mask_len;
	out.original_length = total_text_tokens_to_use;
	return out;
}

}  // namespace

// build_infilling(
//   encoded_input, remaining_space, rng_seed,
//   min_corruption_rate, max_corruption_rate,
//   mean_tokens_per_span, min_mean_tokens_per_span, max_mean_tokens_per_span,
//   max_mask_spans, max_attempts,
//   prompt_toks, paradigm_end_toks, placeholder_bank
// ) -> (status, used_fallback, input_bytes, label_bytes, unused_bytes, masked_count, original_length)
//
// status:
//   0 = ok
//   1 = not_enough_space
static PyObject* build_infilling(PyObject* self, PyObject* args) {
	(void)self;

	PyObject* encoded_input_obj = nullptr;
	int remaining_space = 0;
	unsigned long long rng_seed = 0;
	double min_corruption_rate = 0.0;
	double max_corruption_rate = 0.0;
	double mean_tokens_per_span = 0.0;
	double min_mean_tokens_per_span = 0.0;
	double max_mean_tokens_per_span = 0.0;
	int max_mask_spans = 0;
	int max_attempts = 0;
	PyObject* prompt_toks_obj = nullptr;
	PyObject* paradigm_end_toks_obj = nullptr;
	PyObject* placeholder_bank_obj = nullptr;
	int num_candidates = 1;

	if (!PyArg_ParseTuple(
				args,
				"OiKdddddiiOOO|i",
				&encoded_input_obj,
				&remaining_space,
				&rng_seed,
				&min_corruption_rate,
				&max_corruption_rate,
				&mean_tokens_per_span,
				&min_mean_tokens_per_span,
				&max_mean_tokens_per_span,
				&max_mask_spans,
				&max_attempts,
				&prompt_toks_obj,
				&paradigm_end_toks_obj,
				&placeholder_bank_obj,
				&num_candidates)) {
		return nullptr;
	}

	if (remaining_space <= 0 || max_mask_spans <= 0 || max_attempts <= 0 || mean_tokens_per_span <= 0.0) {
		return Py_BuildValue("iOyyyii", 1, Py_False, "", "", "", 0, 0);
	}
	if (num_candidates < 1) {
		num_candidates = 1;
	}
	if (num_candidates > 16) {
		num_candidates = 16;
	}

	std::vector<int32_t> encoded_input;
	std::vector<int32_t> prompt_toks;
	std::vector<int32_t> paradigm_end_toks;
	if (!seq_to_i32_vector(encoded_input_obj, encoded_input)) {
		return nullptr;
	}
	if (!seq_to_i32_vector(prompt_toks_obj, prompt_toks)) {
		return nullptr;
	}
	if (!seq_to_i32_vector(paradigm_end_toks_obj, paradigm_end_toks)) {
		return nullptr;
	}

	PyObject* placeholder_outer = PySequence_Fast(placeholder_bank_obj, "expected placeholder_bank as a sequence");
	if (!placeholder_outer) {
		return nullptr;
	}
	const Py_ssize_t placeholder_outer_n = PySequence_Fast_GET_SIZE(placeholder_outer);
	if (placeholder_outer_n < max_mask_spans) {
		Py_DECREF(placeholder_outer);
		return Py_BuildValue("iOyyyii", 1, Py_False, "", "", "", 0, 0);
	}

	std::vector<std::vector<int32_t>> placeholder_bank;
	placeholder_bank.reserve(static_cast<size_t>(max_mask_spans));
	PyObject** placeholder_outer_items = PySequence_Fast_ITEMS(placeholder_outer);
	for (int i = 0; i < max_mask_spans; i++) {
		std::vector<int32_t> ph;
		if (!seq_to_i32_vector(placeholder_outer_items[i], ph)) {
			Py_DECREF(placeholder_outer);
			return nullptr;
		}
		placeholder_bank.push_back(std::move(ph));
	}
	Py_DECREF(placeholder_outer);

	const int n_tokens = static_cast<int>(encoded_input.size());
	const int prompt_len = static_cast<int>(prompt_toks.size());

	// Match the Python objective's small buffer.
	const int buffer_tokens = 16;
	const int max_text_budget = remaining_space - prompt_len - buffer_tokens;
	const int max_n_tokens_to_process = std::max(0, std::min(n_tokens, max_text_budget));
	if (max_n_tokens_to_process <= 0) {
		return Py_BuildValue("iOyyyii", 1, Py_False, "", "", "", 0, 0);
	}

	std::vector<CandidateResult> results(static_cast<size_t>(num_candidates));
	std::vector<std::thread> threads;
	threads.reserve(static_cast<size_t>(std::max(0, num_candidates - 1)));
	std::atomic<bool> cancel(false);

	int selected = 0;
	try {
		ScopedGILRelease nogil;

		for (int i = 1; i < num_candidates; i++) {
			const std::uint64_t seed_i = derive_candidate_seed(static_cast<std::uint64_t>(rng_seed), i);
			threads.emplace_back([&, i, seed_i]() {
				try {
					results[static_cast<size_t>(i)] = run_infilling_candidate(
						encoded_input,
						remaining_space,
						seed_i,
						min_corruption_rate,
						max_corruption_rate,
						mean_tokens_per_span,
						min_mean_tokens_per_span,
						max_mean_tokens_per_span,
						max_mask_spans,
						max_attempts,
						prompt_toks,
						paradigm_end_toks,
						placeholder_bank,
						max_n_tokens_to_process,
						&cancel);
				} catch (...) {
					results[static_cast<size_t>(i)].ok = false;
				}
			});
		}

		try {
			results[0] = run_infilling_candidate(
				encoded_input,
				remaining_space,
				derive_candidate_seed(static_cast<std::uint64_t>(rng_seed), 0),
				min_corruption_rate,
				max_corruption_rate,
				mean_tokens_per_span,
				min_mean_tokens_per_span,
				max_mean_tokens_per_span,
				max_mask_spans,
				max_attempts,
				prompt_toks,
				paradigm_end_toks,
				placeholder_bank,
				max_n_tokens_to_process,
				&cancel);
		} catch (...) {
			results[0].ok = false;
		}

		if (results[0].ok && !results[0].used_fallback) {
			cancel.store(true, std::memory_order_relaxed);
		} else {
			for (int i = 1; i < num_candidates; i++) {
				std::thread& t = threads[static_cast<size_t>(i - 1)];
				if (t.joinable()) {
					t.join();
				}
				if (results[static_cast<size_t>(i)].ok && !results[static_cast<size_t>(i)].used_fallback) {
					selected = i;
					cancel.store(true, std::memory_order_relaxed);
					break;
				}
			}
		}

		for (std::thread& t : threads) {
			if (t.joinable()) {
				t.join();
			}
		}
	} catch (const std::exception& exc) {
		cancel.store(true, std::memory_order_relaxed);
		for (std::thread& t : threads) {
			if (t.joinable()) {
				t.join();
			}
		}
		PyErr_SetString(PyExc_RuntimeError, exc.what());
		return nullptr;
	} catch (...) {
		cancel.store(true, std::memory_order_relaxed);
		for (std::thread& t : threads) {
			if (t.joinable()) {
				t.join();
			}
		}
		PyErr_SetString(PyExc_RuntimeError, "unknown error in build_infilling");
		return nullptr;
	}

	const CandidateResult& chosen = results[static_cast<size_t>(selected)];
	if (!chosen.ok) {
		return Py_BuildValue("iOyyyii", 1, Py_False, "", "", "", 0, 0);
	}

	PyObject* input_bytes = i32_vector_to_bytes(chosen.input);
	PyObject* label_bytes = i32_vector_to_bytes(chosen.label);
	PyObject* unused_bytes = i32_vector_to_bytes(chosen.unused);
	if (!input_bytes || !label_bytes || !unused_bytes) {
		Py_XDECREF(input_bytes);
		Py_XDECREF(label_bytes);
		Py_XDECREF(unused_bytes);
		return nullptr;
	}
	return Py_BuildValue(
		"iONNNii",
		0,
		chosen.used_fallback ? Py_True : Py_False,
		input_bytes,
		label_bytes,
		unused_bytes,
		chosen.masked_count,
		chosen.original_length);
}

static PyMethodDef Methods[] = {
	{"build_infilling", build_infilling, METH_VARARGS, "Fast infilling sample builder."},
	{nullptr, nullptr, 0, nullptr},
};

static struct PyModuleDef Module = {
	PyModuleDef_HEAD_INIT,
	"_infilling_fast",
	"Optional C++ backend for Birdie infilling objective.",
	-1,
	Methods,
};

PyMODINIT_FUNC PyInit__infilling_fast(void) {  // NOLINT
	return PyModule_Create(&Module);
}
