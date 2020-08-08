import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def to_var(x, use_gpu=False):
        if use_gpu:
            return Variable(torch.from_numpy(x).cuda())
        else:
            return Variable(torch.from_numpy(x))

model_name = './checkpoint/VM_transe_04Aug.ckpt'

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "./VM/", 
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	#bern_flag = 1, 
	#filter_flag = 1, 
	#neg_ent = 25,
	#neg_rel = 0
)

# dataloader for test
#test_dataloader = TestDataLoader("./dummy/", "link")

# define the model
transe = TransE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 200, 
	p_norm = 1, 
	norm_flag = True)


# define the loss function
model = NegativeSampling(
	model = transe, 
	loss = MarginLoss(margin = 5.0),
	batch_size = train_dataloader.get_batch_size()
)

'''
	Uncomment the following lines to run the trainer. A model has already been pretrained
	and is currently included in thecheckpoint folder.
'''
# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 1000, alpha = 1.0, use_gpu = True)
#trainer.run()
#transe.save_checkpoint(model_name)


'''
	Generates a dictionary which transforms entities and relations into their corresponding integer ids
	Also generates two dictionaries which map ids to their corresponding entity or relation.
'''
# load the files and then create a mapping from entities and relations to id
entid_lines = []
relid_lines = []
with open("./VM/entity2id.txt", "r") as f:
    entid_lines = f.read().split('\n')[1:] # skip first line because it's not relevent
with open("./VM/relation2id.txt", "r") as f:
    relid_lines = f.read().split('\n')[1:]

ent_rel_id = {
    x.split(' ')[0] : int(x.split(' ')[1]) for x in entid_lines + relid_lines
}

id_to_ent = {
    int(x.split(' ')[1]) : x.split(' ')[0] for x in entid_lines
}

id_to_rel = {
	int(x.split(' ')[1]) : x.split(' ')[0] for x in relid_lines
}

''
# load the checkpoint
transe.load_checkpoint(model_name)

'''
	The following code blocks contain various preliminary TransE tests on the Veronica Mars knowledge graph.
	Note that the following code is poorly written since we were rapidly experimenting.
	We will comment out every block of test code and uncomment them as you wish.

	Note that the top scoring predictions are not output in any particular order.
'''


'''
# predict top 5 friends for each of the characters in this list
characters = [
    'Veronica_Mars', 'Duncan_Kane', 'Weevil_Navarro', 'Wallace_Fennel', 'Keith_Mars', 'Jake_Kane', 'Meg_Manning'
]
friend_of_id = ent_rel_id['friend_of']

for c in characters:
	data = {
	    'batch_h': np.array([ent_rel_id[c]]),
	    'batch_t': np.array([x for x in range(len(entid_lines))]),
	    'batch_r': np.array([friend_of_id]),
	    'mode': 'tail_batch'
	}
	res = transe.predict({
	    'batch_h': to_var(data['batch_h']),
	    'batch_t': to_var(data['batch_t']),
	    'batch_r': to_var(data['batch_r']),
	    'mode': data['mode']
	})
	friend_index = np.argpartition(res,5)
	friend_index2 = np.argpartition(res,-5)
	friends = [id_to_ent[x] for x in friend_index[:5]]
	friends2 = [id_to_ent[x] for x in friend_index2[-5:]] # use to compare bottom 5 scores to top 5 scores
	print("Friends of {}: ".format(c), end='')
	for f in friends:
		print(f, end=', ')
	print()
print()

# predict top 5 dislikes per character
dislikes_id = ent_rel_id['dislikes']
for c in characters:
	data = {
	    'batch_h': np.array([ent_rel_id[c]]),
	    'batch_t': np.array([x for x in range(len(entid_lines))]),
	    'batch_r': np.array([dislikes_id]),
	    'mode': 'tail_batch'
	}
	res = transe.predict({
	    'batch_h': to_var(data['batch_h']),
	    'batch_t': to_var(data['batch_t']),
	    'batch_r': to_var(data['batch_r']),
	    'mode': data['mode']
	})
	anti_friend_index = np.argpartition(res,5)
	anti_friends = [id_to_ent[x] for x in anti_friend_index[:5]]
	print("Characters disliked by {}: ".format(c), end='')
	for f in anti_friends:
		print(f, end=', ')
	print()
print()
'''

'''
# predict top 3 clues for each case
caseIds = ["Case{}".format(x) for x in range(1, 20) if x != 7] # Case 7 has no entry for some reason -- bug in training generation
clue_of_id = ent_rel_id['clue_of']
best_clues = []
for c in caseIds:
	data = {
	    'batch_h': np.array([x for x in range(len(entid_lines))]),
	    'batch_t': np.array([ent_rel_id[c]]),
	    'batch_r': np.array([clue_of_id]),
	    'mode': 'head_batch'
	}
	res = transe.predict({
	    'batch_h': to_var(data['batch_h']),
	    'batch_t': to_var(data['batch_t']),
	    'batch_r': to_var(data['batch_r']),
	    'mode': data['mode']
	})
	top_x = 10
	clue_index = np.argpartition(res,top_x)
	for x in clue_index[:top_x]:
		if x not in best_clues:
			best_clues.append(x)
	clues = [id_to_ent[x] for x in clue_index[:top_x]]
	print("Best clues of {}: ".format(c), end='')
	for clue in clues:
		print(clue, end=', ')
	print()
print()
'''


'''
	TSNE visualisation of the TransE embedding for the case clues
'''
'''
num_entities = len(entid_lines)
num_rel = len(relid_lines)
e_embeddings = []
names = []

# List of clues that we want to visualise
# use wanted_clues or best_clues in the visualisation
wanted_clues = [
	"SK", "Carrie's_Grade", "Case14", "Rolling_Stones_music", "win_of_District_Extemporaneous_Speaking_Competition", "Chuck_Rook", "black,_silk_sheets", "diary", "house_key", "texts",
	"Case8", "Meg_Manning's_Purity", "Club_Leadership", "purity_test_score", "Grind_Girl_Magazine", "Meg_Manning", "Manning", "diary",
	"Abel_Koonz's_confession", "Crime_Photographs", "Abel_Koonz's_bloody_clothing", "backpack", "shot_glass", "Lilly's_Secret", "Phone_call", "shoes", "white_sneakers", "tapes",
	"Credit_Card_Purchases", "Speeding_Ticket", "Mystery_Number", "Diamond_Pendant", "Neptune_Grand_Hotel", "Leticia_Navarro", "Caitlin_Ford", "Case2", "shot_glass", "Motorcycle_Jobs",
	"Bathroom_breaks", "Duncan's_Lie", "Sun_Tea", "Ghetto_Brew", "Bedroom", "Pizza_tip", "Drug_problem", "Jack_Daniels", "Case10", "Soccer_uniform"
]
wanted_clues_id = [ent_rel_id[x] for x in wanted_clues]

# clue visualisation
for i in best_clues:
	emb = transe.ent_embeddings(to_var(np.array(i)))
	e_embeddings.append(emb.detach().numpy())
	names.append(id_to_ent[i])


e_embeddings = np.array(e_embeddings)
tsne_result = TSNE(n_components=2).fit_transform(e_embeddings)
df = pd.DataFrame()
df['tsne-2d-one'] = tsne_result[:,0]
df['tsne-2d-two'] = tsne_result[:,1]
print(tsne_result.shape)

plt.scatter(tsne_result[:,0], tsne_result[:,1])
for i in range(len(names)):
	plt.text(tsne_result[i,0], tsne_result[i, 1], names[i], fontsize=6)
plt.savefig('all_clue_emb.png')
'''

'''
	The following block saves the trained TransE embedding to a file called VM_TransE_emb.txt
'''
'''
# Save embeddings to file
to_file = ""
# I feel like this is probably an inefficient way of doing this procedure 
# because of all the string copies
for e_id in id_to_ent:
	name = id_to_ent[e_id]
	emb = transe.ent_embeddings(to_var(np.array(e_id)))
	new_line = name
	for v in emb.detach().numpy():
		new_line += ' {}'.format(v)
	to_file += "{}\n".format(new_line)
for r_id in id_to_rel:
	name = id_to_rel[r_id]
	emb = transe.rel_embeddings(to_var(np.array(r_id)))
	new_line = name
	for v in emb.detach().numpy():
		new_line += ' {}'.format(v)
	to_file += "{}\n".format(new_line)
to_file = to_file[:-1] # strip trailing new line 
with open("VM_TransE_emb.txt", "w") as f:
	f.write(to_file)
'''

'''
	Predict the most likely perpatrators
'''
'''
data = {
	'batch_h': np.array([ent_rel_id['Perpetrator']]),
	'batch_t': np.array([x for x in range(len(entid_lines))]),
	'batch_r': np.array([ent_rel_id['described_as']]),
	'mode': 'tail_batch'
}
res = transe.predict({
	'batch_h': to_var(data['batch_h']),
	'batch_t': to_var(data['batch_t']),
	'batch_r': to_var(data['batch_r']),
	'mode': data['mode']
})
perp_index = np.argpartition(res,25)
perps = [id_to_ent[x] for x in perp_index[:25]]
print("Top 25 Perps @ 50% included:\n{}".format(", ".join(perps)))
print()
'''

'''
	Test financial status prediction
'''
'''
# these people's financial status were not included in training
removed_fin = [
    ['Lilly_Kane', 'upper_class'],
    ['Mandy', 'lower_class'],
    ["Casey's_grandmother", 'upper_class'],
    ["Jim_Cho", "lower_class"],
    ["Chardo_Navarro", "lower_class"],
    ["Duncan_Kane", "upper_class"],
    ["Mr_Gant", "upper_class"],
    ["Lianne_Mars", "lower_class"],
    ["Weevil_Navarro", "lower_class"],
    ["Troy_Vandegraff", "upper_class"],
    ["Mrs_Gant", "upper_class"],
    ["Logan_Echolls", "upper_class"],
    ["Celeste_Kane", "upper_class"],
    ["Jake_Kane", "upper_class"]
]

num_correct = 0
for person, fin_stat in removed_fin:
	data = {
		'batch_h': np.array([ent_rel_id[person]]),
		'batch_t': np.array([140, 160]), # 160 -> lower_class, 140 -> upper_class
		'batch_r': np.array([ent_rel_id['has_financial_status']]),
		'mode': 'tail_batch'
	}
	res = transe.predict({
		'batch_h': to_var(data['batch_h']),
		'batch_t': to_var(data['batch_t']),
		'batch_r': to_var(data['batch_r']),
		'mode': data['mode']
	})
	print(res)
	best_fin_stats_i = np.argpartition(res,1) 
	best_fin_stats = 'lower_class' if best_fin_stats_i[0] == 0 else 'upper_class'

	pred_fin_stat = id_to_ent[np.argmin(res)]
	correct = "Correct" if pred_fin_stat == fin_stat else "Incorrect"
	num_correct += fin_stat == best_fin_stats
	print("{} actual Financial Status: {}.".format(person, fin_stat))
	print(best_fin_stats)
	print()
print("Fin stat accuracy: {}".format(num_correct/len(removed_fin)))
'''
