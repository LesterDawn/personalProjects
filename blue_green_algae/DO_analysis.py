import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

file_dir = "./datasets/jinji_lake/"
j_all = pd.read_csv(file_dir + 'processed_j-all.csv')
j_all_new = j_all[j_all['phycoprotein'] <= 5000].reset_index().drop('index', axis=1)

j_phyco_1800 = j_all_new[j_all_new['phycoprotein'] <= 1800].reset_index().drop('index', axis=1)
print(spearmanr(j_phyco_1800['phycoprotein'], j_phyco_1800['dissolved oxygen'])[0])
j_phyco_5000 = j_all_new[j_all_new['phycoprotein'] > 1800].reset_index().drop('index', axis=1)
print(spearmanr(j_phyco_5000['phycoprotein'], j_phyco_5000['dissolved oxygen'])[0])

# plt.scatter(j_all_new['phycoprotein'], j_all_new['dissolved oxygen'], s=1)
# plt.xlabel('phycocyanin (cells/L)')
# plt.ylabel('dissolved oxygen (mg/L)')
# plt.show()

file_dir = "./datasets/dushu_lake/"
d_all = pd.read_csv(file_dir + 'processed_d-all.csv')
d_all_new = d_all[d_all['dissolved oxygen'] > 2].reset_index().drop('index', axis=1)
print(spearmanr(d_all_new['phycoprotein'], d_all_new['dissolved oxygen'])[0])
plt.scatter(d_all_new['phycoprotein'], d_all_new['dissolved oxygen'], s=1)
plt.xlabel('phycocyanin (cells/L)')
plt.ylabel('dissolved oxygen (mg/L)')
plt.show()
