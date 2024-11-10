from api_tokenizer import APITokenizer
from transformers import AutoTokenizer
import copy
import numpy as np

class Tokenizer(APITokenizer):
	def __init__(self, model_name:str=None):
		"""Initializes a HuggingFace tokenizer with a specific model or a default one."""
		super().__init__()

		if model_name is None:
			model_name = "togethercomputer/LLaMA-2-7B-32K"
		self.model_name = model_name

		try:
			self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
		except Exception as e:
			print(f"Failed to load tokenizer for model {self.model_name}: {e}")
			raise e

	def encode(self, text):
		"""Encodes a given text into tokens."""
		try:
			return self.tokenizer.encode(text)
		except Exception as e:
			print(f"Failed to encode text: {e}")
			raise e

	def decode(self, token_ids):
		"""Decodes token ids back to text."""
		try:
			return self.tokenizer.decode(token_ids)
		except Exception as e:
			print(f"Failed to decode tokens: {e}")
			raise e

	def __call__(self, text):
		"""Processes input text or a list of texts using the encode method."""
		if isinstance(text, str):
			return self.encode(text)
		elif isinstance(text, list):
			return [self.encode(t) for t in text]
		else:
			raise TypeError("Input must be a string or a list of strings.")

class Tokenizer(APITokenizer):
	def __init__(
			self,
			huggingface_name=None,
			name=None,
			add_eos_token=False,
			add_bos_token=False,
			bos_token="<s>",
			eos_token="</s>",
			pad_token="<pad>",
			sep_token="<sep>",
			bos_token_id=31494,
			# bos_token_id=1,
			eos_token_id=2,
			pad_token_id=0,
			sep_token_id=32001-2,
			vocab_size=32000,
			**kwargs,):
		super().__init__()
		
		bos_token_id=31494
		sep_token_id=31999
		vocab_size=32000
		eos_token_id = 2

		self.infilling_sentinel_token_id_start = 31900
		self.infilling_sentinel_token_id_end = 32000

		self.autoencoding_sentinel_token_id_start = 31900
		self.autoencoding_sentinel_token_id_end = 32000


		# Makes chat tag easier read during debugging, particularly with packed sequences
		self.special_matches = [
			("</s>", "█",),
		]

		if huggingface_name is None:
			huggingface_name = "togethercomputer/LLaMA-2-7B-32K"
		self.huggingface_name = huggingface_name

		
		vocabulary = AutoTokenizer.from_pretrained(self.huggingface_name)
		
		self.name = name
		self.tokenizer = vocabulary
		self.vocab_size = vocab_size
		self.bos_token_id = bos_token_id
		self.eos_token_id = eos_token_id
		self.pad_token_id = pad_token_id
		self.sep_token_id = sep_token_id
		self.bos_token = bos_token
		self.eos_token = eos_token
		self.pad_token = pad_token
		self.sep_token = sep_token
		self.add_eos_token = add_eos_token
		self.add_bos_token = add_bos_token

		self.pad_id = pad_token_id
		self.bos_id = bos_token_id
		self.eos_id = eos_token_id
		self.sep_id = sep_token_id

		trs = [self.tokenizer.decode([idx]).strip()[0] for idx in range(31900, 32000)]

		# Replace unused tokens with emojis for easier debugging
		trs = "书构米连操装和ぐ反̌仮员昭ശ兴客删මවპċഷသᵉ居타𝓝थ現ˇ종助唐瀬ន微１Ġほ舞내중Ē导效방ḏ深梅料월每洲회茶败ഞểヨ些双嘉모바ษ進음ญ丁故計遠교재候房명两ფ才합止番ɯ奇怪联역泰백ὀげべ边还黃왕收弘给书构米连操装和ぐ反̌仮员昭ശ兴客删මවპċഷသᵉ居타𝓝थ現ˇ종助唐瀬ន微１Ġほ舞내중Ē导效방ḏ深梅料월每洲회茶败ഞểヨ些双嘉모바ษ進음ญ丁"
		decode_emojis = "🌐🌋🗻🏠🏡⛪️🏢🏣🏤🏥🏦🏨🏩🏪🏫🏬🏭🏯🏰💒🗼🗽🗾⛲️⛺️🌁🌃🌄🌅🌆🌇🌉♨️🌌🎠🎡🎢💈🎪🎭🎨🎰🚂🚃🚄🚅🚆🚇🚈🚉🚊🚝🚞🚋🚌🚍🚎🚏🚐🚑🚒🚓🚔🚕🚖🚗🚘🚙🚚🚛🚜🚲⛽️🚨🚥🚦🚧⚓️⛵️🚣🚤🚢✈️💺🚁🚟🚠🚡🚀🚪⏳⌚️⏰🕓🌑🌓🌕🌚🌝🌞🌠☁️⛅️🌀🌈🌂☔️⚡️❄️⛄️🔥💧🌊🎃🎄🎆🎇✨🎈🎉🎊🎋🎌🎍🎎🎏🎐🎑🎀🎁🎫⚽️⚾️🏀🏈🏉🎾🎱🎳⛳️🎣🎽🎿🏂🏄🏇🏊🚴🚵🏆🎯🎮🎲⬛️⬜️🔶🔷🔸🔹🔺🔻💠🔘🔲🔳⚪️⚫️🔴🔵🇨🇳🇩🇪🇪🇸🇫🇷🇬🇧🇮🇹🇯🇵🇰🇷🇷🇺🇺🇸"
		decode_emojis = "🌐🌋🗻🏠🎮⛪️🏢🏣🏤🏥🏦🏨🏩🏪🏫🏬🏭🏯🏰💒🗼🗽🗾⛲️⛺️🌁🌃🌄🌅🌆🌇🌉♨️🌌🎠🎡🎢💈🎪🎭🎨🎰🚂🚃🚄🔶🚆🚇🚈🚉🚊🚝🚞🚋🚌🚍🔷🚏🚐🚑🚒🚓🚔🚕🚖🚗🚘🚙🚚🚛🚜🚲⛽️🚨🚥🚦🚧⚓️⛵️🚣🚤🚢✈️💺🚁🚟🚠🚡🚀🚪⏳⌚️⏰🕓🌑🌓🌕🌚🌝🌞🌠☁️⛅️🌀🌈🌂☔️⚡️❄️⛄️🔥💧🌊🎃🎄🎆🎇✨🎈🎉🎊🎋🎌🎍🎎🎏🎐🎑🎀🎁🎫⚽️⚾️🏀🏈🏉🎾🎱🎳⛳️🎣🎽🎿🏂🏄🏇🏊🚴🚵🏆🎯"
		
		self.special_matches_decoding = []
		for tr, emoji in zip(trs, decode_emojis):
			self.special_matches_decoding.append((tr, emoji))


		self.special_matches_decoding = [
			[0,'书','🌐',],
			[1,'构','🌋',],
			[2,'米','🗻',],
			[3,'连','🏠',],
			[4,'操','🎮',],
			[5,'装','⛪',],
			[6,'和','🇬🇧',],
			[7,'ぐ','🏢',],
			[8,'反','🏣',],
			[9,'反̌ '[1:],'🏤',],
			[10,'仮','🏥',],
			[11,'员','🏦',],
			[12,'昭','🏨',],
			[13,'ശ','🏩',],
			[14,'兴','🏪',],
			[15,'客','🏫',],
			[16,'删','🏬',],
			[17,'ම','🏭',],
			[18,'ව','🏯',],
			[19,'პ','🏰',],
			[20,'ċ','💒',],
			[21,'ഷ','🗼',],
			[22,'သ','🗽',],
			[23,'ᵉ','🗾',],
			[24,'居','⛲',],
			[25,'타','🇮🇹',],
			[26,'𝓝','⛺',],
			[27,'थ','👽',],
			[28,'現','🌁',],
			[29,'ˇ','🌃',],
			[30,'종','🌄',],
			[31,'助','🌅',],
			[32,'唐','🌆',],
			[33,'瀬','🌇',],
			[34,'ន','🌉',],
			[35,'微','♨',],
			[36,'１','🇯🇵',],
			[37,'Ġ','🌌',],
			[38,'ほ','🎠',],
			[39,'舞','🎡',],
			[40,'내','🎢',],
			[41,'중','💈',],
			[42,'Ē','🎪',],
			[43,'导','🎭',],
			[44,'效','🎨',],
			[45,'방','🎰',],
			[46,'ḏ','🚂',],
			[47,'深','🚃',],
			[48,'梅','🚄',],
			[49,'料','🔶',],
			[50,'월','🚆',],
			[51,'每','🚇',],
			[52,'洲','🚈',],
			[53,'회','🚉',],
			[54,'茶','🚊',],
			[55,'败','🚝',],
			[56,'ഞ','🚞',],
			[57,'ể','🚋',],
			[58,'ヨ','🚌',],
			[59,'些','🚍',],
			[60,'双','🔷',],
			[61,'嘉','🚏',],
			[62,'모','🚐',],
			[63,'바','🚑',],
			[64,'ษ','🚒',],
			[65,'進','🚓',],
			[66,'음','🚔',],
			[67,'ญ','🚕',],
			[68,'丁','🚖',],
			[69,'故','🚗',],
			[70,'計','🚘',],
			[71,'遠','🚙',],
			[72,'교','🚚',],
			[73,'재','🚛',],
			[74,'候','🚜',],
			[75,'房','🚲',],
			[76,'명','⛽',],
			[77,'两','🎃',],
			[78,'ფ','🚨',],
			[79,'才','🚥',],
			[80,'합','🚦',],
			[81,'止','🚧',],
			[82,'番','⚓',],
			[83,'ɯ','🇺🇸',],
			[84,'奇','⛵',],
			[85,'怪','🇬🇪',],
			[86,'联','🚣',],
			[87,'역','🚤',],
			[88,'泰','🚢',],
			[89,'백','✈',],
			[90,'ὀ','🧟',],
			[91,'げ','💺',],
			[92,'べ','🚁',],
			[93,'边','🚟',],
			[94,'还','🚠',],
			[95,'黃','🚡',],
			[96,'왕','🚀',],
			[97,'收','🚪',],
			[98,'弘','⏳',],
			[99,'给','⌚',],
			[100,'书','🇦🇬',],
			[101,'构','⏰',],
			[102,'米','🕓',],
			[103,'连','🌑',],
			[104,'操','🌓',],
			[105,'装','🌕',],
			[106,'和','🌚',],
			[107,'ぐ','🌝',],
			[108,'反','🌞',],
			[109,'ǧ','🇺🇦',],
			[110,'仮','☁',],
			[111,'员','🪖',],
			[112,'昭','⛅',],
			[113,'ശ','🩸',],
			[114,'兴','🌀',],
			[115,'客','🌈',],
			[116,'删','🌂',],
			[117,'ම','☔',],
			[118,'ව','🌻',],
			[119,'პ','⚡',],
			[120,'ċ','🪦',],
			[121,'ഷ','❄',],
			[122,'သ','🤍',],
			[123,'ᵉ','⛄',],
			[124,'居','🎖️',],
			[125,'타','🔥',],
			[126,'𝓝','💧',],
			[127,'थ','🌊',],
			[128,'現','🎃',],
			[129,'ˇ','🎄',],
			[130,'종','🎆',],
			[131,'助','🎇',],
			[132,'唐','✨',],
			[133,'瀬','🎈',],
			[134,'ន','🎉',],
			[135,'微','🎊',],
			[136,'１','🎋',],
			[137,'Ġ','🎌',],
			[138,'ほ','🎍',],
			[139,'舞','🎎',],
			[140,'내','🎏',],
			[141,'중','🎐',],
			[142,'Ē','🎑',],
			[143,'导','🎀',],
			[144,'效','🎁',],
			[145,'방','🎫',],
			[146,'ḏ','⚽',],
			[147,'深','🛸',],
			[148,'梅','⚾',],
			[149,'料','🧠',],
			[150,'월','🏀',],
			[151,'每','🏈',],
			[152,'洲','🏉',],
			[153,'회','🎾',],
			[154,'茶','🎱',],
			[155,'败','🎳',],
			[156,'ഞ','⛳',],
			[157,'ể','💙',],
			[158,'ヨ','🎣',],
			[159,'些','🎽',],
			[160,'双','🎿',],
			[161,'嘉','🏂',],
			[162,'모','🏄',],
			[163,'바','🏇',],
			[164,'ษ','🏊',],
			[165,'進','🚴',],
			[166,'음','🚵',],
			[167,'ญ','🏆',],
			[168,'丁','🎯',],

			[169,'ҡ','🇸🇪',],
			# [170,''ྱ','🇨🇦',],
		]
	
	def orca_encode(self, x):
		if isinstance(x, list):
			return [self.orca_encode(e) for e in x]
		for special_matches_idx, (query, value) in enumerate(self.special_matches):
			x = x.replace(query, value)
		return self.encode(x)
	
	# ## allow for any other function to go through self.vocab
	# def __getattr__(self, name):
	# 	return getattr(self.vocabulary, name)
	
	def __getattr__(self, name):
		if hasattr(self.vocabulary, name):
			return getattr(self.vocabulary, name)
		else:
			raise AttributeError(f"'{type(self.vocabulary).__name__}' object has no attribute '{name}'")

	def __deepcopy__(self, memo):
		cls = self.__class__
		result = cls.__new__(cls)
		memo[id(self)] = result
		
		result.huggingface_name = copy.deepcopy(self.huggingface_name, memo)
		result.name = copy.deepcopy(self.name, memo)
		result.vocab_size = self.vocab_size
		result.bos_token_id = self.bos_token_id
		result.eos_token_id = self.eos_token_id
		result.pad_token_id = self.pad_token_id
		result.sep_token_id = self.sep_token_id
		result.add_bos_token = self.add_bos_token
		result.add_eos_token = self.add_eos_token

		result.tokenizer = self.tokenizer
		return result
		
	def tokenize(self, input_str):
		return self.tokenizer.tokenize(input_str)

	def from_pretrained(self, *args, **kwargs):
		return LlamaTokenizerFast(*args, **kwargs)
	
	def encode(self, input_str):
		if not len(input_str):
			return np.int32([])
		
		encoded = self.tokenizer.encode(input_str)

		if isinstance(input_str, list):
			if self.add_bos_token:
				for e in encoded:
					e.insert(0, self.bos_token_id)
			if self.add_eos_token:
				for e in encoded:
					e.append(self.eos_token_id)
			return [(e) for e in encoded]
		else:
			if self.add_bos_token:
				encoded.insert(0, self.bos_token_id)
			if self.add_eos_token:
				encoded.append(self.eos_token_id)
			return (encoded)
		
	def __call__(self, input_str):
		return self.encode(input_str)
	
	
	def _decode(self, input_ids):
		if input_ids is None:
			return ""
		try:
			if len(input_ids) == 0:
				return ""
		except:
			pass
		if isinstance(input_ids, np.ndarray):
			input_ids = input_ids.tolist()
		if isinstance(input_ids, int):
			input_ids = [input_ids,]
		if isinstance(input_ids[0], list):
			return [self.decode(e) for e in input_ids]
		
		our_str = self.tokenizer.decode(input_ids)

		for special_matches_idx, (idx, query, value) in enumerate(self.special_matches_decoding):
			# our_str = our_str.replace(value, query)
			our_str = our_str.replace(query, value)

		return our_str

	
	def decode(self, input_ids):
		try:
			return self._decode(input_ids)
		except:
			try:
				return self._decode(np.int32(input_ids))
			except:
				return [self._decode(np.int32(e)) for e in input_ids]
	
	
	def batch_decode(self, input_ids):
		if isinstance(input_ids, np.ndarray):
			input_ids = input_ids.tolist()
		return [self.decode(e) for e in input_ids]
		



if __name__ == "__main__":
	tokenizer = Tokenizer()
	print(f"tokenizer: {tokenizer}")

	text = "What color is the sky?"
	encoded = tokenizer.encode(text)
	print(f"encoded: {encoded}")


