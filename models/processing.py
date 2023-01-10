import pandas as pd
import numpy as np
from stimuli import *

vocab = pd.read_csv("../data/vocab.csv")

embeddings = pd.read_csv("../data/swow_associative_embeddings.csv").transpose().values
#reduces the vocab to exclude short words, words with spaces, and capitalized words
# new_vocab, new_embeddings = boards.reduce_vocab_embeddings(vocab, embeddings)
# print("new vocab length=",len(new_vocab))

# random.seed(20)
# saves targets to targets.csv inside the "data" folder
# targets = boards.select_wordpairs(new_vocab, new_embeddings, 0.7, 1)
# print("targets created!")

# word1 = "cat"
# word2 = "lion"

# boards.compute_similarity(word1, word2, vocab, embeddings)

# boards.create_final_board('../data', new_embeddings, new_vocab, 17)
# print("boards created!")

with open('../data/boards.json') as json_file:
    final_boards = json.load(json_file)

# wp = "debit-check"
# combs = RSA.compute_board_combos(wp, final_boards)
# wp_index = list(combs.wordpair).index(wp)

# x = RSA.pragmatic_speaker(wp, embeddings, list(vocab.Word), vocab, final_boards, beta=200)

# # need to get the ordered list of pragmatic clues
# candidate_probs = x[wp_index]
# sorted_prob_indices = np.argsort(candidate_probs)[::-1]
# sorted_words = [list(vocab.Word)[i] for i in sorted_prob_indices]
# print(f"top 10 clues for {wp}=", sorted_words[:10])
# pd.DataFrame(sorted_words).to_csv('../data/literal.csv')



### RANDOM WALK CODE ###
swow = SWOW('../data')

swow.get_final_clues(vocab, embeddings, final_boards, walk_steps = 8)

# ## note that word1 and word2 MUST be in targets.csv for the code below to run
# word1 = "debit"
# word2 = "check"
# # the code below will save the total number of times each clue has been visited 
# # in the union for a certain number of steps (budget_list) inside the walk_data folder
# swow.union_candidates(word1,word2,8, vocab)

# # the code below will save an example walk and its union/intersection words for a certain number of steps (budget_list)
# # inside the walk_data folder
# swow.get_example_walk(word1, word2, [16])

# swow.save_candidates(budget_list=[4])

# with open('../data/walk_data/union_candidates.json') as json_file:
#     union_candidates = json.load(json_file)

# swow.choose_candidates(union_candidates)