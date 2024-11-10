#include <iostream>
#include <vector>
#include <random>

extern "C" {
    int update_rng_seed(int seed);
    int get_span_length(int mean_span_width);
    double get_corruption_percentage(int corrupt_chance);
    bool should_we_corrupt_this_span(float corrupt_chance, int proposed_span_length, int span_start, int last_corrupted_location, int last_span_width);
    bool weiter(int span_start, int array_length, int corrupted_input_length, int labels_length, int max_allowed_length);
    // std::vector<int>& get_list();
    int run_span_corruption(int* input_array, int* out_inputs_array, int* out_labels_array, int* results_array, int input_length, int max_allowed_length, int inputs_length, int labels_length, float mean_corrupt_chance, int mean_span_width, int sentinel_start, int sentinel_end);
    int run_ssm_span_corruption(int* input_array, int* out_inputs_array, int* out_labels_array, int* results_array, int input_length, int max_allowed_length, int inputs_length, int labels_length, float mean_corrupt_chance, int mean_span_width, int sentinel_start, int sentinel_end);
    void set_not_ready();
    void set_ready();
    int get_is_ready();
    int is_ready = 1;
    int get_random_in_range(int min_val, int max_val);
}

// static std::default_random_engine generator;

static std::default_random_engine generator(12345);  // Seeded with a fixed value for determinism

int update_rng_seed(int seed) {
    generator.seed(seed);
    return 0;
}

int get_span_length(int mean_span_width) {
    /*
        - Generates a span length using a normal distribution centered around mean_span_width with a standard deviation of 1.0.
        - Ensures the span length is at least 1.
    */

    std::normal_distribution<double> distribution(mean_span_width, 1.0);
    int span_length = static_cast<int>(distribution(generator));
    return std::max(1, span_length);
}

double get_corruption_percentage(int corrupt_chance) {
    /*
        - Generates a corruption percentage using a normal distribution centered around corrupt_chance with a standard deviation of 1.0.
        - Ensures this percentage is within 50% to 200% of the corrupt_chance.
    */
    std::normal_distribution<double> distribution(corrupt_chance, 1.0);
    double current_corruption_percentage = distribution(generator);
    current_corruption_percentage = std::max(corrupt_chance * 0.5, current_corruption_percentage);
    current_corruption_percentage = std::min(corrupt_chance * 2.0, current_corruption_percentage);
    return current_corruption_percentage;
}

bool should_we_corrupt_this_span(float corrupt_chance, int proposed_span_length, int span_start, int last_corrupted_location, int last_span_width) {
    /*
        - Decides whether to corrupt a span based on corrupt_chance, proposed_span_length, and span_start.
        - Uses a uniform distribution to generate a random value and checks if it's less than the scaled corrupt_chance.
        - Ensures the end of the proposed span is positive (i.e., within the bounds of the input array).
    */

    //    new_min_loc is minimum of 1
   if (last_span_width != -1) {
        int new_min_loc = (last_corrupted_location + ((int) last_span_width / 3) + 1);
        new_min_loc = std::max(1, new_min_loc);
        if (span_start <= new_min_loc) {
            return false;
        }
   }

    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    auto val = distribution(generator);
    bool c0 = (val * (float) proposed_span_length) < corrupt_chance;
    bool c1 = 0 < (span_start + proposed_span_length);
    // std::cout << "corrupt_chance: " << corrupt_chance << std::endl;
    // std::cout << "c0: " << c0 << std::endl;
    // std::cout << "c1: " << c1 << std::endl;
    // std::cout << "val: " << val << std::endl;
    // std::cout << std::endl;
    return c0 && c1;
}

int get_random_in_range(int min_val, int max_val) {
    /*
        - Decides whether to corrupt a span based on corrupt_chance, proposed_span_length, and span_start.
        - Uses a uniform distribution to generate a random value and checks if it's less than the scaled corrupt_chance.
        - Ensures the end of the proposed span is positive (i.e., within the bounds of the input array).
    */
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    auto val = distribution(generator);

    int random_value = (int) (val * (max_val - min_val)) + min_val;
    return random_value;
}

bool weiter(int span_start, int array_length, int corrupted_input_length, int labels_length, int max_allowed_length) {
    /*
        - Determines whether the corruption process should continue based on the current span_start, lengths of the corrupted_input and labels, and the max_allowed_length.
        - Ensures that
            (the current position is within the input array)
            and
            (the combined length of the corrupted input and labels does not exceed the maximum allowed length minus one).
    
    */
    bool c0 = span_start < array_length;
    bool c1 = (corrupted_input_length + labels_length) < max_allowed_length-1;
    return c0 && c1;
}

std::vector<int>& get_list() {
    // make and return a new list
    return *(new std::vector<int>());


    // static std::vector<int> list;
    // return list;
}

void freeListMemory(std::vector<int>& list) {
    std::vector<int>().swap(list);
}

void set_not_ready() {
    is_ready = 0;
}

void set_ready() {
    is_ready = 1;
}

int get_is_ready() {
    return is_ready;
}


int run_span_corruption(int* input_array, int* out_inputs_array, int* out_labels_array, int* results_array, 
    int inputs_length, int labels_length,
    int input_length, int max_allowed_length, float mean_corrupt_chance, int mean_span_width, int sentinel_start, int sentinel_end) {

    set_not_ready();


    auto corrupted_input = get_list();
    auto labels = get_list();

    labels.clear();
    corrupted_input.clear();

    // int span_start = -(mean_span_width * 4);
    int span_start = 0;
    int sentinel_offset = 0;
    int num_not_corrupted = 0;
    int num_corrupted = 0;
    int last_corrupted_location = 0;
    int last_span_width = -1;


    while (weiter(span_start, input_length, static_cast<int>(corrupted_input.size()), static_cast<int>(labels.size()), max_allowed_length)) {
        int max_allowable_span_length = (max_allowed_length - static_cast<int>(corrupted_input.size()) - static_cast<int>(labels.size()) - 2);

        if (max_allowable_span_length <= 0) {
            break;
        }

        int proposed_span_length = get_span_length(mean_span_width);
        // std::cout << "mean_corrupt_chance: " << mean_corrupt_chance << std::endl;
        bool should_corrupt = should_we_corrupt_this_span(mean_corrupt_chance, proposed_span_length, span_start, last_corrupted_location, last_span_width);
        // std::cout << "should_corrupt: " << should_corrupt << std::endl;

        if (should_corrupt) {
            // int sentinel_idx = (sentinel_start + sentinel_offset);

            // sentinel_idx is random between sentinel_start and sentinel_end
            int sentinel_idx = get_random_in_range(sentinel_start, sentinel_end);
            int span_end = (span_start + proposed_span_length);
            last_corrupted_location = span_end;
            last_span_width = proposed_span_length;

            if (span_end >= 0) {
                corrupted_input.push_back(sentinel_idx);
                int real_start = std::max(0, span_start);

                int was_seen = 0;
                for (int i = real_start; i < span_end; ++i) {
                    if (i < input_length) {
                        // corrupted_input.push_back(input_array[i]);
                        labels.push_back(input_array[i]);
                        was_seen = 1;
                        num_corrupted += 1;
                    }
                }
                if (was_seen == 1) {
                    labels.push_back(sentinel_idx);
                    // num_corrupted += (span_end - real_start);
                }
                span_start = span_end;
                // Nudge it forward 1
                // span_start = (span_end + 1);
            }
            else {
                span_start += proposed_span_length;
            }

            sentinel_offset = (sentinel_offset + 1) % (sentinel_end - sentinel_start);
        }
        else {
            if (span_start >= 0) {
                corrupted_input.push_back(input_array[span_start]);
                num_not_corrupted += 1;
            }
            span_start += 1;
        }
    }

    int i_max;
    i_max =  static_cast<int>(labels.size());
    i_max = std::min(labels_length, i_max);
    for (int i = 0; i < i_max; ++i) {
        out_labels_array[i] = labels[i];
    }

    i_max = static_cast<int>(corrupted_input.size());
    i_max = std::min(inputs_length, i_max);
    for (int i = 0; i < i_max; ++i) {
        out_inputs_array[i] = corrupted_input[i];
    }

    results_array[0] = static_cast<int>(corrupted_input.size());
    results_array[1] = static_cast<int>(labels.size());
    results_array[2] = num_not_corrupted;
    results_array[3] = num_corrupted;

    // clean up pointers
    // freeListMemory(corrupted_input);
    // freeListMemory(labels);


    set_ready();
    return span_start;





}

int run_ssm_span_corruption(int* input_array, int* out_inputs_array, int* out_labels_array, int* results_array, 
    int inputs_length, int labels_length,
    int input_length, int max_allowed_length, float mean_corrupt_chance, int mean_span_width, int sentinel_start, int sentinel_end) {

    set_not_ready();


    auto corrupted_input = get_list();
    auto labels = get_list();

    labels.clear();
    corrupted_input.clear();

    // int span_start = -(mean_span_width * 4);
    int span_start = 0;
    int sentinel_offset = 0;
    int num_not_corrupted = 0;
    int num_corrupted = 0;
    int last_corrupted_location = 0;
    int last_span_width = -1;


    while (weiter(span_start, input_length, static_cast<int>(corrupted_input.size()), static_cast<int>(labels.size()), max_allowed_length)) {
        int max_allowable_span_length = (max_allowed_length - static_cast<int>(corrupted_input.size()) - static_cast<int>(labels.size()) - 2);

        if (max_allowable_span_length <= 0) {
            break;
        }

        int proposed_span_length = get_span_length(mean_span_width);
        // std::cout << "mean_corrupt_chance: " << mean_corrupt_chance << std::endl;
        bool should_corrupt = should_we_corrupt_this_span(mean_corrupt_chance, proposed_span_length, span_start, last_corrupted_location, last_span_width);
        // std::cout << "should_corrupt: " << should_corrupt << std::endl;

        if (should_corrupt) {
            // int sentinel_idx = (sentinel_start + sentinel_offset);
            int sentinel_idx = get_random_in_range(sentinel_start, sentinel_end);
            int span_end = (span_start + proposed_span_length);
            last_corrupted_location = span_end;
            last_span_width = proposed_span_length;

            if (span_end >= 0) {
                corrupted_input.push_back(sentinel_idx);
                int real_start = std::max(0, span_start);

                int was_seen = 0;
                for (int i = real_start; i < span_end; ++i) {
                    if (i < input_length) {
                        // corrupted_input.push_back(input_array[i]);
                        // labels.push_back(input_array[i]);
                        was_seen = 1;
                        num_corrupted += 1;
                    }
                }
                // if (was_seen == 1) {
                //     labels.push_back(sentinel_idx);
                //     // num_corrupted += (span_end - real_start);
                // }
                // span_start = span_end;
                // Nudge it forward 1
                span_start = (span_end + 1);
            }
            else {
                span_start += proposed_span_length + (int) (get_span_length(mean_span_width) * (1/3));
            }

            sentinel_offset = (sentinel_offset + 1) % (sentinel_end - sentinel_start);
        }
        else {
            if (span_start >= 0) {
                corrupted_input.push_back(input_array[span_start]);
                num_not_corrupted += 1;
            }
            span_start += 1;
        }
    }

    int i_max;
    // i_max =  static_cast<int>(labels.size());
    // i_max = std::min(labels_length, i_max);
    // for (int i = 0; i < i_max; ++i) {
    //     out_labels_array[i] = labels[i];
    // }
    for (int i = 0; i < input_length; ++i) {
        out_labels_array[i] = input_array[i];
    }

    i_max = static_cast<int>(corrupted_input.size());
    i_max = std::min(inputs_length, i_max);
    for (int i = 0; i < i_max; ++i) {
        out_inputs_array[i] = corrupted_input[i];
    }

    results_array[0] = static_cast<int>(corrupted_input.size());
    // results_array[1] = static_cast<int>(labels.size());
    results_array[1] = input_length;
    results_array[2] = num_not_corrupted;
    results_array[3] = num_corrupted;

    // clean up pointers
    // freeListMemory(corrupted_input);
    // freeListMemory(labels);


    set_ready();
    return span_start;





}