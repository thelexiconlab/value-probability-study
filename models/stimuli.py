import random
from random import randrange
import os
import json
import pickle
import itertools
import walker
import math
import warnings
import scipy.spatial.distance
from heapq import nlargest
import pandas as pd
import numpy as np
import networkx as nx
import math
from collections import defaultdict
from scipy.special import softmax
from nltk import edit_distance

class boards:

    '''
    main restrictions for ALL words
      (1) excluding any compound words
      (2) excluding words with length < 2 or >15
      (3) excluding words that are capitalized, or have non-alphabetical characters in them
      (4) excluding taboo words
    
    additional restrictions for distractors
      (1) remove w1 and w2 
      (2) remove words that have w1 or w2 in them (e.g., ground vs. underground)
      (3) remove words that are within edit distance of 3 or less from the targets
      (4) remove words that are already targets or distractors
    
    additional restrictions for clues
      (1) exclude distractors
      (2) exclude any clues that are within short edit distances of the distractors
      (3) exclude any clues that are words on any of the boards
      
    '''

    def reduce_vocab_embeddings(vocab, embeddings):
      '''
      applying some corrections for ALL words/vocab:
      (1) excluding any compound words
      (2) excluding words with length < 2 or >12
      (3) excluding words that are capitalized
      (4) excluding taboo words
      '''

      all_words = list(vocab.Word)

      main_words = [w for w in all_words if (len(w) < 15) & (len(w) > 2) & (' ' not in w) & (w[0].islower()) & (w.isalpha())]
      taboo_words = pd.read_csv("../data/taboo.csv").word_list.values
      
      main_words = [w for w in main_words if w not in taboo_words]
      #print(set(all_words) ^ set(main_words))

      main_word_indices = [all_words.index(w) for w in main_words]
      new_vocab = vocab.loc[main_word_indices]
      new_embeddings = embeddings[main_word_indices]
      return new_vocab, new_embeddings
    
    def exclusions_for_distractors_clues(w1, w2, initial_list):
      #print(f"for {w1} and {w2}, and list of {len(initial_list)} ")

      # remove w1 and w2 from initial_list
      final_list = [w for w in initial_list if w not in [w1,w2]]
      #print(f"reduced to {len(final_list)}")
      # also remove words that have w1 or w2 in them (e.g., ground vs. underground)
      final_list = [w for w in final_list if (w1 not in w) & (w2 not in w) & (w not in w1) & (w not in w2)]
      # also remove words that are within edit distance of 3 or less from the targets
      final_list = [w for w in final_list if (edit_distance(w1,w)>3) & (edit_distance(w2,w)>3)]
      # also remove words that are targets or distractors
      targets = pd.read_csv('../data/targets.csv')
      targets = list(targets.Word1) + list(targets.Word2) + list(targets.distractor)
      #print("not in:", targets)
      final_list = [w for w in final_list if w not in targets]
      
      return final_list

    def exclude_current_clues(candidate_list, current_clues):
      '''
      excluding all clues and variants of those clues so far from the list
      '''
      after_subwords_list1 = []
      subwords_list2 = []

      if len(current_clues)>0:
        final_list = [w for w in candidate_list if w not in current_clues]
        print("size after excluding currentcclues is=", len(final_list))
        
        ## first look for final_list words in current_clues
        ## this tests if a candidate like "bug" is in "bugs"
        for w in final_list:
          if not any(w in c for c in current_clues):
            after_subwords_list1+= [w]
        
        # then look for current_clues in after_subwords_list1
        ## this tests if a clue like "bug" is in a candidate like "bugs"
        for w in after_subwords_list1:
          if any(c in w for c in current_clues):
            # if any of the current_clues are found as part of any of the words in after_subwords_list1
            # add to a new list
            subwords_list2+= [w]
        
        # now exclude subword_list2 from after_subwords_list1
        after_subwords_list1 = [w for w in after_subwords_list1 if w not in subwords_list2]
        print("size after excluding subwords of currentcclues is=", len(after_subwords_list1))
        return after_subwords_list1
      else:
        print("current clues is empty, size of list=", len(candidate_list))
        return candidate_list
    
    def exclusions_for_clues(w1, w2, candidate_list):

      print("size before exclusions is=", len(candidate_list))

      # remove w1 and w2 from initial_list
      final_list = [w for w in candidate_list if w not in [w1,w2]]
      print("size after excluding w1w2 is=", len(final_list))
      #print(f"reduced to {len(final_list)}")
      # also remove words that have w1 or w2 in them (e.g., ground vs. underground)
      final_list = [w for w in final_list if (w1 not in w) & (w2 not in w) & (w not in w1) & (w not in w2)]
      print("size after excluding subwords of w1w2=", len(final_list))
      # also remove words that are within edit distance of 3 or less from the targets
      final_list = [w for w in final_list if (edit_distance(w1,w)>3) & (edit_distance(w2,w)>3)]
      print("size after excluding edit dist w1w2=", len(final_list))
      # also remove words that are targets or distractors
      targets = pd.read_csv('../data/targets.csv')
      targets = list(targets.Word1) + list(targets.Word2) + list(targets.distractor)
      #print("not in:", targets)
      final_list = [w for w in final_list if w not in targets]
      print("size after excluding other targets and distractors=", len(final_list))

      ## also remove subwords of distractors and targets?
      after_subwords_list_1 = []
      # first we look for whether any candidate word is a subword of targets/distractors
      # this catches a candidate like "bug" if "bugs" is a target/distractor
      # because we are testing if bug in bugs
      for w in final_list:
        if not any(w in c for c in targets):
          after_subwords_list_1+= [w]

      print("1: size after excluding subwords of targets and distractors=", len(after_subwords_list_1))

      # next we want to look for whether any target/distractor is a subword of the remaining candidates
      # this catches a distractor like "recipes" if "recipe" is a target/distractor
      # because we are testing if recipe in recipes
      subwords_list_2 = []
      for c in after_subwords_list_1:
        if any(w in c for w in targets ):
          #print(f"found {c}, will need to exclude")
          subwords_list_2+= [c]
      
      after_subwords_list_1 = [w for w in after_subwords_list_1 if w not in subwords_list_2]

      print("2: size after excluding subwords of targets and distractors=", len(after_subwords_list_1))
      ## remove taboo words + extra long words

      main_words = [w for w in after_subwords_list_1 if (len(w) < 15) & (len(w) > 2) & (' ' not in w) & (w[0].islower()) & (w.isalpha())]
      taboo_words = pd.read_csv("../data/taboo.csv").word_list.values
      
      main_words = [w for w in main_words if w not in taboo_words]
      print("size after excluding taboo=", len(main_words))

      ## need asome additional exclusions where board words are also not included

      with open('../data/boards.json') as json_file:
        final_boards = json.load(json_file)
      
      blist = final_boards.values()
      board_words = list(itertools.chain(*blist))

      reduced_words = [w for w in main_words if w not in board_words]

      print("size after excluding board=", len(reduced_words))
      return reduced_words
    
    def exclusions_for_wordpairs(wordpair_list):
      '''
      wordpair_list is a list of lists [[w1,w2], [w1,w2]]
      '''
      target_df = pd.read_csv("../data/targets.csv")
      for index,row in target_df.iterrows():
        w1 = row["Word1"]
        w2 = row["Word2"]
        # remove w1 and w2 from initial_list
        final_list = [w for w in wordpair_list if w not in [w1,w2]]
        # also remove word pairs that have w1 or w2 in them (e.g., ground vs. underground)
        final_list = [[i,j] for i,j in final_list if (w1 not in i) & (w2 not in i) & (i not in w1) & (i not in w2) & (w1 not in j) & (w2 not in j) & (j not in w1) & (j not in w2)]
        # also remove word pairs that are within edit distance of 3 or less from the targets
        final_list = [[i,j] for i, j in final_list if (edit_distance(w1,i)>3) & (edit_distance(w2,i)>3) & (edit_distance(w1,j)>3) & (edit_distance(w2,j)>3)]
        # also remove words that are targets or distractors
        targets = list(target_df.Word1) + list(target_df.Word2) + list(target_df.distractor)
        final_list = [[i,j] for i,j in final_list if (i not in targets) & (j not in targets) ]
      
      return final_list  

    def select_wordpairs(vocab, embeddings, similarity_threshold, n):
      # first compute similarity matrix
      sim_matrix = 1 - scipy.spatial.distance.cdist(embeddings, embeddings, 'cosine')
      # find all indices where similarity is > threshold and less than 0.90 
      greater = np.argwhere((sim_matrix > similarity_threshold) & (sim_matrix < 0.90)).tolist()
      # random sample of list
      wordpair_indices = random.sample(greater, n+200)
      # once we have the indices, we need to convert to words
      words = [[list(vocab.Word)[i], list(vocab.Word)[j]] for i, j in wordpair_indices]
      
      # reduce this to our acceptable word list 
      print(f"original {len(words)}")
      
      targets = pd.read_csv('../data/targets.csv')
      current_n = len(targets)
      while(current_n <= n):
        final_list = boards.exclusions_for_wordpairs(words)
        print(f"reduced to {len(final_list)}")
        print(f"sampling the {current_n}+1 th item")
        final_words = random.sample(final_list, 1)
        df = pd.DataFrame(final_words, columns = ['Word1', 'Word2'])
        
        new_target_df = pd.read_csv("../data/targets.csv")
        final_target_df = pd.concat([new_target_df, df])
        final_target_df.to_csv('../data/targets.csv', index=False)
        
        targets = pd.read_csv('../data/targets.csv')
        current_n = len(targets)
        
    
    def compute_similarity(word1, word2, vocab, embeddings):
      '''
      given two words, a vocabulary and an underlying embedding space for that vocbulary, computes cosine similarity
      '''
      w1_vec = embeddings[list(vocab.Word).index(word1)]
      w1_vec = w1_vec.reshape((1, embeddings.shape[1]))
      w2_vec = embeddings[list(vocab.Word).index(word2)]
      w2_vec = w2_vec.reshape((1, embeddings.shape[1]))
      similarity = 1 - scipy.spatial.distance.cdist(w1_vec, w2_vec, 'cosine')
      print(f"similarity between {word1} and {word2}=", similarity)
      return similarity

    def generate_random_board(words,n, board_words):
        '''
        given a list of words, generates a random sample of n words, excluding the words that are already in board_words
        '''
        flattened_board_words = list(itertools.chain(*board_words))
        # print("len flattened =",len(flattened_board_words))
        # print("flattened_board_words=",flattened_board_words)
        
        reduced_words = list(set(words) - set(flattened_board_words))
        #print("reduced board candidates=", len(reduced_words))
        
        board_sample = random.sample(reduced_words, n)
        return board_sample

    def generate_distractor(w1, w2, embeddings, vocabulary, distance, board_words):
        '''
        given 2 words, generates a close distractor based on underlying semantic embeddings using midpoint 
        '''

        w1_vec = embeddings[list(vocabulary.Word).index(w1)]
        w2_vec = embeddings[list(vocabulary.Word).index(w2)]
        midpoint = (w1_vec + w2_vec)/2
        midpoint = midpoint.reshape((1, embeddings.shape[1]))
        similarities = 1 - scipy.spatial.distance.cdist(midpoint, embeddings, 'cosine')
        y = np.array(similarities)
        y_sorted = np.argsort(-y).flatten() ## gives sorted indices
        closest_words = [list(vocabulary.Word)[i] for i in y_sorted]
        
        final_list = boards.exclusions_for_distractors_clues(w1, w2, closest_words)

        # board words that are already chosen also need to be excluded

        final_list = [w for w in final_list if w not in board_words]
        
        # take random word from top 5-10
        distractor = random.sample(final_list[distance:distance+10], 1)

        ## also return all other words that are NOT close

        far_words = final_list[1000:]
        return distractor, far_words
    
    def create_final_board(data_path, embeddings, vocab, n):
        '''
        saves a board for w1 & w2, 
        but we need to make sure there is nothing else on the board that has similarity over a threshold to 
        the wordpair or distractor
        '''
        final_target_df = pd.DataFrame()
        target_df = pd.read_csv(f"{data_path}/targets.csv".format())
        target_df["wordpair"]= target_df["Word1"]+ "-"+target_df["Word2"]
        all_words = list(target_df.Word1) + list(target_df.Word2)
        final_boards = {}
        # keeps track of which words have been used already and excludes those from future boards

        for index, row in target_df.iterrows():
            w1 = row["Word1"]
            w2 = row["Word2"]
            wordpair = row["wordpair"]

            new_target_df = pd.DataFrame({ 'Word1':[w1] , 'Word2': [w2] ,'wordpair': [wordpair]})
            
            distractor, far_words = boards.generate_distractor(w1, w2, embeddings, vocab, 50, all_words)
            new_target_df["distractor"] = distractor
            final_target_df = pd.concat([final_target_df, new_target_df])
            final_target_df.to_csv('../data/targets.csv', index=False)
            print(f"distractor for {wordpair} is {distractor}")

            # print("far words length=", len(far_words))
            
            # print("all_words=",all_words)
            
            b = boards.generate_random_board(far_words, n, [all_words + distractor]) 
            
            twenty = b + distractor + [w1,w2]
              
            all_words = all_words + twenty
        
            final_boards[wordpair] = twenty
            print("board is=", twenty)
        
        with open('../data/boards.json', 'w') as f:
            json.dump(final_boards, f)   
      
    def boardjson_to_csv(path):
      '''
      takes in a path for boards.json and converts it to a pandas dataframe
      '''
      df = pd.read_json(path).T
      df.reset_index(inplace=True)

      df['board'] = df[df.columns[1:21]].apply(lambda x: ', '.join(x), axis = 1).to_list()
      df["newboard"] = (df["board"].str.replace('"', "").apply(lambda x: ", ".join(f"'{word}'" for word in x.split(", "))))
      df[['index', 'newboard']].to_csv('../data/boards.csv', index=False)
    

class RSA:
  def __init__(self):
    '''
    initialize with embeddings & board matrices
    '''
    with open('../data/boards.json') as json_file:
      self.final_boards = json.load(json_file)
    
    self.vocab = pd.read_csv("../data/vocab.csv")
    self.embeddings = pd.read_csv("../data/swow_associative_embeddings.csv").transpose().values
    self.candidates = list(self.vocab.Word)
    self.create_all_boards_matrices()
  
  def compute_board_combos(self,board_name):
    '''
    inputs:
    (1) board_name ("e1_board1_words")
    output:
    all pairwise combinations of the words on the board

    '''
    board = self.final_boards[board_name]
    all_possible_combs = list(itertools.combinations(board, 2))
    combs_df = pd.DataFrame(all_possible_combs, columns =['Word1', 'Word2'])
    combs_df["wordpair"] = combs_df["Word1"] + '-'+ combs_df["Word2"]
    return combs_df

  def create_board_matrix(self,combs_df, context_board, embeddings, vocab, candidates):
    '''
    inputs:
    (1) combs_df: all combination pairs from a given board
    (2) context_board: the specific board ("lion-tiger")
    (3) embeddings: embedding space to consider
    (4) the vocab over which computations are occurring
    (5) candidates over which the board matrix needs to be computed

    output:
    product similarities of given vocab to each wordpair
    '''
    print("inside create_board_matrix")
    print("context_board=",context_board)
    # grab subset of words in given board and their corresponding glove vectors
    board_df = vocab[vocab['Word'].isin(context_board)]
    board_word_indices = list(board_df.index)
    board_words = board_df["Word"]
    board_vectors = embeddings[board_word_indices]

    # need to obtain embeddings of candidate set

    candidate_index = [list(vocab["Word"]).index(w) for w in candidates]

    candidate_embeddings = embeddings[candidate_index]

    ## clue_sims is the similarity of ALL clues in full candidate space to EACH word on board (size 20)
    clue_sims = 1 - scipy.spatial.distance.cdist(board_vectors, candidate_embeddings, 'cosine')

    ## once we have the similarities of the clue to the words on the board
    ## we define a multiplicative function that maximizes these similarities
    board_df.reset_index(inplace = True)

    ## next we find the product of similarities between c-w1 and c-w2 for that specific board's 190 word-pairs
    ## this gives us a 190 x N array of product similarities for a given combs_df
    ## specifically, for each possible pair, pull out
    f_w1_list =  np.array([clue_sims[board_df[board_df["Word"]==row["Word1"]].index.values[0]]
                          for  index, row in combs_df.iterrows()])
    f_w2_list =  np.array([clue_sims[board_df[board_df["Word"]==row["Word2"]].index.values[0]]
                          for  index, row in combs_df.iterrows()])

    # result is of length 190 for the product of similarities (i.e. how similar each word i is to BOTH in pair)
    # note that cosine is in range [-1, 1] so we have to convert to [0,1] for this conjunction to be valid
    return ((f_w1_list + 1) /2) * ((f_w2_list + 1)/2)

  def create_all_boards_matrices(self):
    '''
    creates the matrix of similarities for all boards and all possible "candidates": currently full vocab
    '''
    print("inside create_all_boards_matrices")
    self.board_combos = {board_name : self.compute_board_combos(board_name) for board_name in self.final_boards.keys()}
    print("board_combos created")

    self.board_matrices = {
      board_name : self.create_board_matrix(self.board_combos[board_name], self.final_boards[board_name], self.embeddings, self.vocab, self.candidates)
            for board_name in self.final_boards.keys()
    }

  def literal_guesser(self, board_name, beta):
    '''
    inputs are:
    (1) board name ("e1_board1_words"),
    (2) representation: embedding space to consider, representations
    (3) modelname: 'glove'
    (4) candidates (a list ['apple', 'mango'] etc.)

    output:
    softmax likelihood of different wordpairs under a given set of candidates

    '''    
    print("inside literal guesser")
    
    boardmatrix = self.board_matrices[board_name]
    return softmax(beta*boardmatrix, axis=0)

  def pragmatic_speaker(self, board_name, beta):
    '''
    inputs:
    (1) board name ("e1_board1_words")
    (2) beta: optimized parameter
    
    outputs:
    softmax likelihood of each possible clue in "candidates"

    '''
    print("inside prag speaker")
    literal_guesser_prob = self.literal_guesser(board_name, beta)
    return softmax(beta*literal_guesser_prob, axis = 1)
  
  def pragmatic_guesser(self,board_name, beta):
    print("inside prag guesser")
    return softmax(beta*(self.pragmatic_speaker(board_name, beta)), axis = 0)
  
  def get_guess_scores(self, beta):
    '''
    obtains literal and pragmatic guesser scores for a given clue/board
    '''
    print("inside guess scores")
    self.clues_df = pd.read_csv(f"../data/clues_final.csv".format())
    guess_scores = pd.DataFrame()

    for wordpair, board in self.final_boards.items():
      print(wordpair)
      # it is possible that the combs has it stored in reverse order
      w1, w2 = wordpair.split("-")
      reverse_wordpair = w2+"-"+w1
      keys = list(self.board_combos.keys())
      wordpairs_in_order = list(self.board_combos[wordpair].wordpair) if wordpair in keys else list(self.board_combos[reverse_wordpair].reverse_wordpair)
      if len(wordpairs_in_order) == 0:
        print("empty, regular:")
        print(self.board_combos[wordpair])
        print("empty, reverse:")
        print(self.board_combos[reverse_wordpair])
        break

      literal_guesser_prob = self.literal_guesser(wordpair, beta)  # this is a 190x12217 array
      prag_guesser_prob = self.pragmatic_guesser(wordpair, beta)
      
      # we need to get prediction scores for specific clues from here

      specific_clue_df = self.clues_df.loc[(self.clues_df["wordpair"]==wordpair)]
      # get indices of these clues & the array corresponding to that clue from literal_guesser_prob
      l_high_a_high_p = literal_guesser_prob[:,self.candidates.index(list(specific_clue_df.high_a_high_p_clue)[0])]
      l_high_a_low_p = literal_guesser_prob[:,self.candidates.index(list(specific_clue_df.high_a_low_p_clue)[0])]
      l_low_a_high_p = literal_guesser_prob[:,self.candidates.index(list(specific_clue_df.low_a_high_p_clue)[0])]
      l_low_a_low_p = literal_guesser_prob[:,self.candidates.index(list(specific_clue_df.low_a_low_p_clue)[0])]

      p_high_a_high_p = prag_guesser_prob[:,self.candidates.index(list(specific_clue_df.high_a_high_p_clue)[0])]
      p_high_a_low_p = prag_guesser_prob[:,self.candidates.index(list(specific_clue_df.high_a_low_p_clue)[0])]
      p_low_a_high_p = prag_guesser_prob[:,self.candidates.index(list(specific_clue_df.low_a_high_p_clue)[0])]
      p_low_a_low_p = prag_guesser_prob[:,self.candidates.index(list(specific_clue_df.low_a_low_p_clue)[0])]
      
      # now we have the array of wordpair scores for each clue, these need to be combined up

      guess_df1 = pd.DataFrame({"guess": wordpairs_in_order })
      guess_df1["clue"] = list(specific_clue_df.high_a_high_p_clue)[0]
      guess_df1["literal_score"] = l_high_a_high_p
      guess_df1["pragmatic_score"] = p_low_a_low_p
      guess_df1["wordpair"] = wordpair

      guess_df2 = pd.DataFrame({"guess": wordpairs_in_order })
      guess_df2["clue"] = list(specific_clue_df.high_a_low_p_clue)[0]
      guess_df2["literal_score"] = l_high_a_low_p
      guess_df2["pragmatic_score"] = p_low_a_low_p
      guess_df2["wordpair"] = wordpair

      guess_df3 = pd.DataFrame({"guess": wordpairs_in_order })
      guess_df3["clue"] = list(specific_clue_df.low_a_high_p_clue)[0]
      guess_df3["literal_score"] = l_low_a_high_p
      guess_df3["pragmatic_score"] = p_low_a_low_p
      guess_df3["wordpair"] = wordpair

      guess_df4 = pd.DataFrame({"guess": wordpairs_in_order })
      guess_df4["clue"] = list(specific_clue_df.low_a_low_p_clue)[0]
      guess_df4["literal_score"] = l_low_a_low_p
      guess_df4["pragmatic_score"] = p_low_a_low_p
      guess_df4["wordpair"] = wordpair
      

      guess_scores = pd.concat([guess_scores, guess_df1, guess_df2, guess_df3, guess_df4])
      guess_scores.to_csv('../data/guess_scores.csv', index = False)





class SWOW:
  def __init__(self, data_path):
    # import target words
    self.target_df = pd.read_csv(f"{data_path}/targets.csv".format())
    self.target_df["wordpair"]= self.target_df["Word1"]+ "-"+self.target_df["Word2"]
    self.target_words = set(self.target_df.Word1).union(self.target_df.Word2)

    self.load_graph(data_path)
    self.index_to_name = {k: v['word'] for k,v in self.graph.nodes(data=True)}
    self.name_to_index = {v['word'] : k for k,v in self.graph.nodes(data=True)}

    # import clues
    self.clues_df = pd.read_csv(f"../data/clues_final.csv".format())
    self.load_random_walks(data_path)
    self.load_clue_walks(data_path)

  def load_graph(self, data_path):
    '''
    reads in networkx graph saved as pickle
    '''
    if os.path.exists(f'{data_path}/walk_data/swow.gpickle') :
      with open(f'{data_path}/walk_data/swow.gpickle', 'rb') as f:
        self.graph = pickle.load(f)
    else :
      self.save_graph(data_path, None)

  def save_graph(self, path, threshold):
    '''
    creates graph directly from pandas edge list and saves to file
    '''
    path = path + '/walk_data/swow_strengths.csv'
    edges = pd.read_csv(path).rename(columns={'R123.Strength' : 'weight'})
    G = nx.from_pandas_edgelist(edges, 'cue', 'response', ['weight'], create_using=nx.DiGraph)
    G = nx.convert_node_labels_to_integers(G, label_attribute = 'word')
    with open('../data/walk_data/swow.gpickle', 'wb') as f:
        pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)

  def load_random_walks(self, data_path):
    '''
    runs n_walks independent random walks of walk_len length from each words
    '''
    if os.path.exists(f'{data_path}/walk_data/walks.pkl'):
      with open(f'{data_path}/walk_data/walks.pkl', 'rb') as f:
        self.rw = pickle.load(f)
    else :
      self.save_random_walks()
  
  def load_clue_walks(self, data_path):
    '''
    runs n_walks independent random walks of walk_len length from each words
    '''
    if os.path.exists(f'{data_path}/walk_data/clue_walks.pkl'):
      with open(f'{data_path}/walk_data/clue_walks.pkl', 'rb') as f:
        self.clues_rw = pickle.load(f)
    else :
      self.save_clue_walks()

  def save_clue_walks(self, n_walks=1000, walk_len=10000):
    '''
    runs n_walks independent random walks of walk_len length from each clue
    '''
    
    self.clue_words = list(set(list(self.clues_df.high_a_high_p_clue) + list(self.clues_df.high_a_low_p_clue) + list(self.clues_df.low_a_high_p_clue) + list(self.clues_df.low_a_low_p_clue)))
    indices = self.get_nodes_by_word(self.clue_words)
    self.clues_rw = walker.random_walks(self.graph, n_walks=n_walks, walk_len=walk_len, start_nodes=indices)
    with open('../data/walk_data/clue_walks.pkl', 'wb') as f:
      pickle.dump(self.clues_rw, f)


  def save_random_walks(self, n_walks = 1000, walk_len = 10000):
    '''
    runs n_walks independent random walks of walk_len length from each words
    '''
    indices = self.get_nodes_by_word(self.target_words)
    self.rw = walker.random_walks(self.graph, n_walks=n_walks, walk_len=walk_len, start_nodes=indices)
    with open('../data/walk_data/walks.pkl', 'wb') as f:
      pickle.dump(self.rw, f)
 
  def chunk(self, l, n):
    '''
    iterates through l in chunks of size n
    '''
    c = itertools.count()
    return (list(it) for _, it in itertools.groupby(l, lambda x: next(c)//n))

  def powers_of_two(self, n):
    '''
    returns all powers of 2 between 2 and n
    '''
    return [2**i for i in range(1, int(math.log(n, 2))+1)]

  def get_nodes_by_word(self, words):
    '''
    looks up which node IDs a list of words correspond to
    '''
    return [self.name_to_index[name] if name in self.name_to_index else None
            for name in words]

  def get_words_by_node(self, nodes):
    '''
    looks up which words a list of node IDs correspond to
    '''
    return [self.index_to_name[index] if index in self.index_to_name else None
            for index in nodes]

  def get_guess_visit_counts(self, clue, board_words, budget_value):
    '''
    returns a visit count of specific guesses on a given board for a given clue
    '''
    clue_indices = self.get_nodes_by_word([clue])
    walks = np.array([x for x in self.clues_rw if x[0] in clue_indices]).tolist()
    guess_counts = defaultdict(int)

    # need to assign 0 count to all words first
    for word in board_words:
      guess_counts[word] = 0
      # now we update these counts based on the walks themselves
      for clue_walk in walks : # for each walk 
        for element in clue_walk[: budget_value]:
          word_list = self.get_words_by_node([element])
          walk_word = word_list[0]
          if walk_word  == word : # if the specific word on the board was visited, then update its count
            guess_counts[word] += 1
    
    guess_count_df = pd.DataFrame.from_dict(guess_counts,orient='index', columns=['visit_count'])
    guess_count_df.reset_index(inplace=True)
    guess_count_df = guess_count_df.rename(columns = {'index':'word'})
    guess_count_df["budget"] = budget_value
    guess_count_df["clue"] = clue

    guess_count_df = guess_count_df.sort_values(by=['visit_count'], ascending=False)
    return guess_count_df
  
  def save_guess_visit_counts(self, budget_list):
    # Loop through specific  clues and boards
    visits = {}
    with open('../data/boards.json') as json_file:
      final_boards = json.load(json_file)

    main_df = pd.DataFrame()
    
    for wordpair, board in  final_boards.items():
      print(wordpair)
      # get the clues for the specific wordpair
      specific_clue_df = self.clues_df.loc[(self.clues_df["wordpair"]==wordpair)]
      
      for budget in budget_list:
        high_a_high_p_df = self.get_guess_visit_counts(list(specific_clue_df.high_a_high_p_clue)[0], board, budget)
        high_a_low_p_df = self.get_guess_visit_counts(list(specific_clue_df.high_a_low_p_clue)[0], board, budget)
        low_a_high_p_df = self.get_guess_visit_counts(list(specific_clue_df.low_a_high_p_clue)[0], board, budget)
        low_a_low_p_df = self.get_guess_visit_counts(list(specific_clue_df.low_a_low_p_clue)[0], board, budget)
        wordpair_df = pd.concat([high_a_high_p_df, high_a_low_p_df,low_a_high_p_df,low_a_low_p_df])
        wordpair_df["wordpair"]= wordpair
        main_df = pd.concat([main_df, wordpair_df])
        main_df.to_csv('../data/guess_visit_counts.csv', index= False)
    
    return main_df
      

  def union_candidates(self, w1, w2, budget_value, vocab):
    '''
    return a list of candidates and their count of visitation up until budget_list steps in the walks stored inside self.rw
    '''

    target_indices = self.get_nodes_by_word([w1, w2])
    walks = np.array([x for x in self.rw if x[0] in target_indices]).tolist()
    union_counts = defaultdict(int)

    # need to assign 0 count to all words first
    for word in list(vocab.Word):
      union_counts[word] = 0
    
    # now we update these counts based on the walks themselves
    for w1_walk, w2_walk in self.chunk(walks, 2) :  
      for element in set(w1_walk[: budget_value]).union(w2_walk[: budget_value]) :
        word_list = self.get_words_by_node([element])
        word = word_list[0]
        union_counts[word] += 1
    
    union_df = pd.DataFrame.from_dict(union_counts,orient='index', columns=['visit_count'])
    union_df.reset_index(inplace=True)
    union_df = union_df.rename(columns = {'index':'word'})

    union_df = union_df.sort_values(by=['visit_count'], ascending=False)#.to_csv('../data/walk_data/union_counts.csv', index= False)

    # with open('../data/walk_data/union_counts.json', 'w') as f:
    #   json.dump(union_counts, f)   

    return union_df
  

  def save_candidates(self, budget_list):
    # Loop through word pairs
    unions = {}
    for w1, w2 in  zip(self.target_df['Word1'], self.target_df['Word2']) :
      print(w1,w2)
      union_counts = self.union_candidates(w1, w2, budget_list)
      union_candidates_list = {budget: sorted(d.items(), key=lambda k_v: k_v[1], reverse=True)
                          for (budget, d) in union_counts.items()}
      union_nodes = {budget: [x[0] for x in d] for (budget, d) in union_candidates_list.items()}
      unions[w1 + '-' + w2] = {'budget=' + str(budget): self.get_words_by_node(d) for (budget, d) in union_nodes.items()}

    with open('../data/walk_data/union_candidates.json', 'w') as f:
      json.dump(unions, f)   

  def choose_candidates(self, walk_candidates):
        '''
        we want to top 2 "accessible" candidates
        '''
        candidates = {}
        for targets in walk_candidates:
            top = random.sample(walk_candidates[targets]['budget=4'][2:10], 2)
            bottom = random.sample(walk_candidates[targets]['budget=4'][-10:],2)
            candidates[targets]=top + bottom
        
        with open('../data/clue_candidates.json', 'w') as f:
            json.dump(candidates, f)
  
  def get_example_walk(self, w1, w2, budget_list):
      target_indices = self.get_nodes_by_word([w1, w2])
      walks = np.array([x for x in self.rw if x[0] in target_indices]).tolist()
      random_index =  randrange(0, 999) # for selecting a random walk from the 1000 walks
      w1_walk = self.get_words_by_node(walks[random_index])
      w2_walk = self.get_words_by_node(walks[random_index+1])

      intersection_counts = {budget : defaultdict(lambda: 0.000001) for budget in budget_list}
      union_counts = {budget : defaultdict(lambda: 0.000001) for budget in budget_list}

      for search_budget in budget_list :
          intersection = list(set(w1_walk[: search_budget]).intersection(w2_walk[: search_budget]))
          intersection_counts[search_budget] = intersection
          union = list(set(w1_walk[: search_budget]).union(w2_walk[: search_budget]))
          union_counts[search_budget] = union

      with open('../data/walk_data/example_intersection.json', 'w') as f:
        json.dump(intersection_counts, f)

      with open('../data/walk_data/example_union.json', 'w') as f:
        json.dump(union_counts, f)

      with open('../data/walk_data/example_walk.json', 'w') as f:
        json.dump({w1_walk[0]:w1_walk[:budget_list[0]], w2_walk[0]: w2_walk[:budget_list[0]]}, f)
  
  def get_final_clues(self, vocab, embeddings, final_boards, walk_steps):
    '''
    this function obtains the visit counts and RSA scores for all potential clues 
    and then identifies the 4 clues
    '''
    # Loop through word pairs
    #final_clues_df = pd.read_csv("../data/clues.csv")

    final_clues_df = pd.DataFrame()

    #current_clues = list(pd.read_csv("../data/current_clues.csv").current_clues)
    current_clues = []

    for index, row in self.target_df[6:7].iterrows():
      w1 = row['Word1']
      w2 = row['Word2']
      distractor = row['distractor']
      wordpair = w1 + "-" + w2
      print("******" ,wordpair, "******")
      # compute a df of visit counts
      union_counts_df = self.union_candidates(row["Word1"], row["Word2"], walk_steps, vocab) 
      print("visit counts complete")
      # compute an np array of scores for different clues in vocab
      combs = RSA.compute_board_combos(wordpair, final_boards)
      wp_index = list(combs.wordpair).index(wordpair)

      x = RSA.pragmatic_speaker(wordpair, embeddings, list(vocab.Word), vocab, final_boards, beta=200)
      print("RSA calculation complete")

      # need to get the ordered list of pragmatic clues
      candidate_probs = x[wp_index]
      candidate_probs = pd.DataFrame(candidate_probs, columns=['pragmatic_score'])
      candidate_probs["word"] = list(vocab.Word)
    
      # need to merge these two together
      final_df = candidate_probs.merge(union_counts_df, on='word', how='left')
      final_df["wordpair"] = wordpair
      final_df["product"] = final_df["visit_count"]*final_df["pragmatic_score"]
      final_df = final_df.sort_values(by="product", ascending = False)

      ## apply exclusions to this set 
      words = list(final_df.word)

      reduced_words = boards.exclusions_for_clues(w1, w2, words)

      # exclude current clues ## we also want to exclude variants of current clues

      final_candidates = boards.exclude_current_clues(reduced_words, current_clues)
      

      print("exclusions complete!")

      final_df = final_df[final_df['word'].isin(final_candidates)]

      final_df = final_df.reset_index()
      final_df.to_csv("../data/final_df.csv", index=False)

      # final_df = pd.read_csv("../data/final_df.csv")
      # final_df = final_df[~final_df['word'].isin(current_clues)]

      # final_df = final_df[final_df['visit_count']!= 0]
     
      # clues_df = pd.DataFrame({'wordpair': [wordpair], 'distractor': distractor})
      
      # # get mean and sd of accessibility
      # threshold_a = final_df["visit_count"].mean() + final_df["visit_count"].std()
      # print("threshold_a=",threshold_a)

      # # get mean and sd of prag score
      # threshold_p = final_df["pragmatic_score"].mean() + final_df["pragmatic_score"].std()
      # print("threshold_p=",threshold_p)
      
      # final_df["over_threshold_a_true"] = final_df["visit_count"] > threshold_a 
      # final_df["over_threshold_p_true"] = final_df["pragmatic_score"] > threshold_p

      # print("sorting by high visit count and high prag score")
      # # but prioritize pragmatics!
      # final_df = final_df.sort_values(by=['pragmatic_score'], ascending=False)
      # high_a_high_p_df = final_df.loc[(final_df["over_threshold_a_true"]==True) & (final_df["over_threshold_p_true"]== True)]
      # print(list(high_a_high_p_df["word"])[0])

      # print("sorting by high visit count and low prag score")
      # final_df = final_df.sort_values(by=['visit_count'], ascending=False)
      # high_a_low_p_df = final_df.loc[(final_df["over_threshold_a_true"]==True) & (final_df["over_threshold_p_true"]== False)]
      # print(list(high_a_low_p_df["word"])[0])

      # print("sorting by high prag score and low visit count")
      # final_df = final_df.sort_values(by=['pragmatic_score'], ascending=[False])
      # low_a_high_p_df = final_df.loc[(final_df["over_threshold_a_true"]==False) & (final_df["over_threshold_p_true"]== True)]
      # print(list(low_a_high_p_df["word"])[0])

      # # defining the low_a_low_p
      # print("defining the low_a_low_p")

      # low_a_low_p_df = final_df.loc[(final_df["over_threshold_a_true"]==False) & (final_df["over_threshold_p_true"]== False)]
      # print(list(low_a_low_p_df["word"])[0])

      # current_clues = current_clues + [list(high_a_high_p_df["word"])[0],list(high_a_low_p_df["word"])[0],list(low_a_high_p_df["word"])[0],list(low_a_low_p_df["word"])[0]]
      # # print("current_clues=",current_clues)
      # # #print("high_a_high_p=",high_a_high_p)
      # clues_df["high_a_high_p_clue"] = list(high_a_high_p_df["word"])[0]
      # clues_df["high_a_high_p_visitcount"] = list(high_a_high_p_df["visit_count"])[0]
      # clues_df["high_a_high_p_prag_score"] = list(high_a_high_p_df["pragmatic_score"])[0]

      # #print("high_a_low_p=",high_a_low_p)
      # clues_df["high_a_low_p_clue"] = list(high_a_low_p_df["word"])[0]
      # clues_df["high_a_low_p_visitcount"] = list(high_a_low_p_df["visit_count"])[0]
      # clues_df["high_a_low_p_prag_score"] = list(high_a_low_p_df["pragmatic_score"])[0]

      # #print("low_a_high_p=",low_a_high_p)
      # clues_df["low_a_high_p_clue"] = list(low_a_high_p_df["word"])[0]
      # clues_df["low_a_high_p_visitcount"] = list(low_a_high_p_df["visit_count"])[0]
      # clues_df["low_a_high_p_prag_score"] = list(low_a_high_p_df["pragmatic_score"])[0]

      # #print("low_a_low_p=",low_a_low_p)
      # clues_df["low_a_low_p_clue"] = list(low_a_low_p_df["word"])[0]
      # clues_df["low_a_low_p_visitcount"] = list(low_a_low_p_df["visit_count"])[0]
      # clues_df["low_a_low_p_prag_score"] = list(low_a_low_p_df["pragmatic_score"])[0]

      # final_clues_df = pd.concat([final_clues_df, clues_df])
      
      # final_clues_df.to_csv('../data/clues_final.csv', index=False)
      # print("clue selection complete")


