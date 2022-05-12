import random
import spacy
import numpy as np
import gender_guesser.detector as gender

class Counterfactual:
	def __init__(self, spacy=None, transformer=None, tokenizer=None, noun_diff=None,
				article_diff=None, articles=[], names_file=None, hardcoded={}):
		self.spacy = spacy
		self.transformer = transformer
		self.tokenizer = tokenizer
		self.names_file = names_file
		self.hardcoded = hardcoded
		self.articles = articles
		self.noun_diff = self.embedding_for_word(noun_diff[0]) - self.embedding_for_word(noun_diff[1])
		if article_diff is None:
			self.article_diff = self.noun_diff
		else:
			self.article_diff = self.embedding_for_word(article_diff[0]) - self.embedding_for_word(article_diff[1])

		self.nam_detect = None
		self.recc_names = {'M': [], 'F': []}

	def _gender_deps(self, sentence):
		pairings = []
		for token in sentence:
			if type(token) == type(''):
				raise Exception('Send a spaCy document, not a string, to _gender_deps !')
			if token.dep_ == "det" or token.dep_ == "amod":
				pairings.append([token, token.head])
		return pairings

	def _flip_noun(self, og_token):
		if type(og_token) == type(''):
			raise Exception('Send a spaCy document, not a string, to _flip_noun !')

		if 'AdvType=Tim' in og_token.tag_: # don't change years
			return None
		elif og_token.pos_ == "PRON":
			diff = 1 * self.noun_diff
		else:
			diff = 0.7 * self.noun_diff

		if og_token.text.lower() in self.hardcoded.keys():
			return self.hardcoded[og_token.text.lower()]
		elif 'Gender=Masc' in og_token.tag_:
			# flip to feminine
			alt_word = self.closest_word(og_token.text, -1 * diff)
		elif (og_token.pos_ == "NOUN") and ("Gender" not in og_token.tag_):
			# el/la lingüista - word stays the same but adj. should change
			# not letting pronouns into here
			alt_word = og_token.text
		else:
			# flip to masculine
			alt_word = self.closest_word(og_token.text, 1 * diff)

		alt_nlp = self.spacy(alt_word)[0]
		if ((alt_nlp.pos_ not in ['NOUN', 'PRON']) or # don't allow change from noun to a verb
			 ('NOUN__Gender=Fem' in og_token.tag_ and 'NOUN__Gender=Fem' in alt_nlp.tag_) or
			 ('NOUN__Gender=Masc' in og_token.tag_ and 'NOUN__Gender=Masc' in alt_nlp.tag_) or
			 ('Number=Sing' in og_token.tag_ and 'Number=Plur' in alt_nlp.tag_) or
			 ('Number=Plur' in og_token.tag_ and 'Number=Sing' in alt_nlp.tag_)): # or
			 #(og_token.lemma_ not in alt_nlp.lemma_)):
			 return None

		return alt_word

	def embedding_for_word(self, word):
		id = self.tokenizer.encode(word)[1]
		return self.transformer.embeddings.word_embeddings.weight[id].detach().numpy()

	def _cosine_similarity(self, vec1, vec2):
		len1 = np.linalg.norm(vec1)
		len2 = np.linalg.norm(vec2)
		dot_product = np.dot(vec1, vec2)
		return dot_product / (len1 * len2)

	def closest_word(self, word, diff):
		original_id = self.tokenizer.encode(word)[1]

		# make the diff adjustment
		encoded = self.embedding_for_word(word)
		new_word = encoded + diff

		mostSim = 0
		leastDistWord = -1
		index = 0

		for word in self.transformer.embeddings.word_embeddings.weight:
			dist = self._cosine_similarity(new_word, word.detach().numpy())
			if (dist > mostSim) and (index > 6) and (index != original_id):
				mostSim = dist
				leastDistWord = index
			index += 1
		return self.tokenizer.decode([leastDistWord])

	def flip_sentence(self, sentence):
		doc = self.spacy(sentence)
		pairings = self._gender_deps(doc)
		words = []
		just_saw_proper_noun = False

		for token in doc:
			alt_word = None

			if (self.names_file is not None) and (token.pos_ == "PROPN") and (not just_saw_proper_noun): # swap first names
				if self.nam_detect is None:
					# initialize only first time it's needed
					# currently still set to Spanish names
					with open(self.names_file, 'r') as nff:
						for name in nff:
							if name[0] == "#":
								continue # readme
							spanish_pop = name[36]
							if spanish_pop != " " and spanish_pop > "3":
								conventional_binary_gender = name.split(' ')[0]
								if conventional_binary_gender in ['M', 'F']:
									name = name.split(' ')[2]
									self.recc_names[conventional_binary_gender].append(name)
					self.nam_detect = gender.Detector()

				conventional_binary_gen = self.nam_detect.get_gender(token.text)
				just_saw_proper_noun = True
				if 'female' in conventional_binary_gen:
					alt_word = random.choice(self.recc_names['M'])
				elif 'male' in conventional_binary_gen:
					alt_word = random.choice(self.recc_names['F'])
				# leave ambiguous or unknown names alone

			else:
				just_saw_proper_noun = False

				if token.pos_ == "NOUN" or token.pos_ == "PRON":
					alt_word = self._flip_noun(token)

				elif len(pairings) > 0 and token.text == pairings[0][0].text:
					diff = self.noun_diff
					if token.text.lower() in self.articles:
						diff = self.article_diff

					dep_noun_token = pairings[0][1]
					if (('PROPN_' not in dep_noun_token.tag_) and ('Gender' in token.tag_)):
						alt_noun = self._flip_noun(dep_noun_token)
						if alt_noun is not None: # don't change ADJ if the noun would not change
							alt_noun_nlp = self.spacy(alt_noun)
							if 'Gender=Masc' in token.tag_:
								alt_word = self.closest_word(token.text, -0.6 * diff)
							else:
								alt_word = self.closest_word(token.text, 0.6 * diff)
					# move onto next paired word
					pairings = pairings[1:]

			if alt_word is None or alt_word.lower() == token.text.lower():
				words.append(token.text)
			else:
				words.append(alt_word)
		return ' '.join(words).replace(' ,', ',').replace(' .', '.').replace('..', '.')