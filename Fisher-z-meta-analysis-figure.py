#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[2]:


import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import joblib

pd.set_option('display.max_columns',100)


# In[3]:


df1 = joblib.load("brenda_all_model_result_293dataset.joblib").reset_index(drop=True)
df1.shape
df1.head(2)
df1.tail(2)


# In[4]:


df2 = joblib.load("sabio-rk_all_model_result_121dataset.joblib").reset_index(drop=True)
df2.shape
df2.head(2)
df2.tail(2)


# In[5]:


df3 = joblib.load("literature_all_model_result_25dataset.joblib").reset_index(drop=True)
df3.shape
df3.head(2)
df3.tail(2)


# In[6]:


sel = ["flag",'RaSP', 'ThermoMPNN','FoldX', 'DDGun', 'Pythia', 'GEMME', 'VESPA', 'ESM1V', 'SaProt', 'ProSST','PSICHIC', 'DynamicBind', 'SMINA', 'Boltz-2', 'DSDP', 'Value']

kcat = pd.concat([df1.loc[df1["flag"] == "brenda_kcat",sel],
                  df2.loc[df2["flag"] == "sabio_kcat",sel],
                  df3.loc[df3["flag"] == "literature_kcat",sel]
                 ])

km = pd.concat([df1.loc[df1["flag"] == "brenda_km",sel],
                df2.loc[df2["flag"] == "sabio_km",sel],
                df3.loc[df3["flag"] == "literature_km",sel]
                 ])

kk = pd.concat([df1.loc[df1["flag"] == "brenda_kk",sel],
                df2.loc[df2["flag"] == "sabio_kk",sel],
                df3.loc[df3["flag"].isin(["literature_kk","literature_normalized_fitness","literature_DMS_score"]),sel]
                 ])


# In[7]:


kcat["n"] = kcat["Value"].apply(lambda x: len(x))
km["n"]   = km["Value"].apply(lambda x: len(x))
kk["n"]   = kk["Value"].apply(lambda x: len(x))


# In[8]:


kcat_corr = kcat.loc[:,["n"]]
km_corr   = km.loc[:,["n"]]
kk_corr   = kk.loc[:,["n"]]

sel = ['RaSP', 'ThermoMPNN','FoldX', 'DDGun', 'Pythia', 'GEMME', 'VESPA', 'ESM1V', 'SaProt', 'ProSST','PSICHIC', 'DynamicBind', 'SMINA', 'Boltz-2', 'DSDP']

for name in sel:
    kcat_corr[name] = kcat.apply(lambda x: spearmanr(x[name],x["Value"])[0],axis=1)
    km_corr[name]   = km.apply(lambda x: spearmanr(x[name],x["Value"])[0],axis=1)
    kk_corr[name]   = kk.apply(lambda x: spearmanr(x[name],x["Value"])[0],axis=1)

kcat_corr.shape    
kcat_corr.head(2)

km_corr.shape
km_corr.head(2)

kk_corr.shape
kk_corr.head(2)


# In[9]:


kcat_corr["DDGun"] = kcat_corr["DDGun"].apply(lambda x: x*(-1))
kcat_corr["VESPA"] = kcat_corr["VESPA"].apply(lambda x: x*(-1))
kcat_corr["SMINA"] = kcat_corr["SMINA"].apply(lambda x: x*(-1))
kcat_corr["Boltz-2"] = kcat_corr["Boltz-2"].apply(lambda x: x*(-1))
kcat_corr["DSDP"] = kcat_corr["DSDP"].apply(lambda x: x*(-1))

kk_corr["DDGun"] = kk_corr["DDGun"].apply(lambda x: x*(-1))
kk_corr["VESPA"] = kk_corr["VESPA"].apply(lambda x: x*(-1))
kk_corr["SMINA"] = kk_corr["SMINA"].apply(lambda x: x*(-1))
kk_corr["Boltz-2"] = kk_corr["Boltz-2"].apply(lambda x: x*(-1))
kk_corr["DSDP"] = kk_corr["DSDP"].apply(lambda x: x*(-1))

km_corr["RaSP"]  = km_corr["RaSP"].apply(lambda x: x*(-1))
km_corr["ThermoMPNN"]  = km_corr["ThermoMPNN"].apply(lambda x: x*(-1))
km_corr["FoldX"] = km_corr["FoldX"].apply(lambda x: x*(-1))
km_corr["Pythia"] = km_corr["Pythia"].apply(lambda x: x*(-1))
km_corr["GEMME"]  = km_corr["GEMME"].apply(lambda x: x*(-1))
km_corr["ESM1V"]  = km_corr["ESM1V"].apply(lambda x: x*(-1))
km_corr["SaProt"] = km_corr["SaProt"].apply(lambda x: x*(-1))
km_corr["ProSST"] = km_corr["ProSST"].apply(lambda x: x*(-1))
km_corr["PSICHIC"] = km_corr["PSICHIC"].apply(lambda x: x*(-1))
km_corr["DynamicBind"] = km_corr["DynamicBind"].apply(lambda x: x*(-1))

kcat_corr.head(2)
km_corr.head(2)
kk_corr.head(2)


# In[10]:


box_list = []
for name in sel:
    cur = [ i for i in kcat_corr[name].tolist()]   + [ i for i in km_corr[name].tolist()]   + [ i for i in kk_corr[name].tolist()]
    box_list.append(cur)
    
col = ['#32037d','#32037d','#32037d','#32037d','#32037d',  '#c94e65','#c94e65','#c94e65','#c94e65','#c94e65','#46c1be','#46c1be','#46c1be','#46c1be','#46c1be']
cur = pd.DataFrame({"model": sel,"corr":box_list,"col":col})
cur["corr_median"] = cur["corr"].apply(lambda x: np.median(x))
cur = cur.sort_values("corr_median",ascending=False)

box1 = cur.copy()
box1
del cur


# In[11]:


import numpy as np
from scipy import stats
from scipy.stats import norm


# In[12]:


def fisher_z_transform(r):

    r = np.clip(r, -0.999999, 0.999999)
    return np.arctanh(r)


def inverse_fisher_z_transform(z):

    return np.tanh(z)


def fisher_z_meta_analysis(r_list, n_list):

    z_list = [fisher_z_transform(r) for r in r_list]

    w_list = [n - 3 for n in n_list]

    z_pooled = np.sum(
        np.array(w_list) * np.array(z_list)
    ) / np.sum(w_list)

    pooled_r = inverse_fisher_z_transform(z_pooled)

    return round(pooled_r, 3)


# In[13]:


tmp = []
for i in sel:
    r_list = kcat_corr[i].tolist() + km_corr[i].tolist() + kk_corr[i].tolist()
    n_list = kcat_corr["n"].tolist() + km_corr["n"].tolist() + kk_corr["n"].tolist()
    tmp.append((i,fisher_z_meta_analysis(r_list, n_list)))
tmp


# In[14]:


cur = pd.DataFrame({"model": [i[0] for i in tmp],"pooled_r":[i[1] for i in tmp],"col": [ '#32037d','#32037d','#32037d','#32037d','#32037d',  '#c94e65','#c94e65','#c94e65','#c94e65','#c94e65','#46c1be','#46c1be','#46c1be','#46c1be','#46c1be']})
cur = cur.sort_values("pooled_r",ascending=False)

box2 = cur.copy()
box2
del cur


# In[15]:


InteractiveShell.ast_node_interactivity = "last"

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# In[16]:


fig, axes = plt.subplots(1, 2, figsize=(10, 4.5),dpi=300)

plt.subplots_adjust(wspace=0.1, hspace=0.9)

bp = axes[0].boxplot(box1["corr"].tolist(),labels=box1["model"].tolist(),showfliers=False,patch_artist=True)       
axes[0].tick_params(axis='x', rotation=90)
axes[0].set_yticks(np.arange(-1, 1.2, 0.25)) 
axes[0].set_ylim((-1.1,1.1)) 
axes[0].set_ylabel("Spearman Corr",fontsize=12, fontweight='bold')   
axes[0].axhline(y=0, color='black', linestyle='--', linewidth=1)
axes[0].text(-0.18, 1.06, "A", transform=axes[0].transAxes, fontsize=14, fontweight='bold', va='top')

for box,color in zip(bp['boxes'],box1["col"].tolist()):
    box.set(facecolor=color, alpha=0.8)
    
for median in bp['medians']:
    median.set(color='black', linewidth=2)
    
axes[1].bar(box2["model"].tolist(),box2["pooled_r"].tolist(),color=box2["col"].tolist())
axes[1].tick_params(axis='x', rotation=90)   
axes[1].set_yticks(np.arange(-0.2, 0.35, 0.1)) 
axes[1].set_ylabel("Fisher z meta-analysis",fontsize=12, fontweight='bold') 
axes[1].text(-0.18, 1.06, "B", transform=axes[1].transAxes, fontsize=14, fontweight='bold', va='top')

for num in [0,1]:
    # 批量设置x/y轴刻度标签加粗
    for label in axes[num].get_xticklabels():
        label.set_fontweight('bold')
    for label in axes[num].get_yticklabels():
        label.set_fontweight('bold')      
    for spine in axes[num].spines.values():
        spine.set_color('black')    
        spine.set_linewidth(1.5) 

plt.tight_layout()
plt.savefig("fig06ab20260415.png")
plt.show()


# In[17]:


InteractiveShell.ast_node_interactivity = "all"


# In[18]:


sel = ["flag", 'GEMME', 'VESPA', 'ESM1V', 'SaProt', 'ProSST','UniKP_Kcat_Km_Kcat/Km',
       'EITLEM-Kinetics_Kcat_Km_Kcat/Km', 'CataPro_Kcat_Km_Kcat/Km',
       'CatPred_Kcat_Km', 'DLKcat_Kcat', 'Value']

kcat = pd.concat([
                  df3.loc[df3["flag"] == "literature_kcat",sel]
                 ])
kcat["UniKP"] = kcat["UniKP_Kcat_Km_Kcat/Km"].apply(lambda x: [float(i[0]) for i in x])
kcat["EITLEM-Kinetics"] = kcat["EITLEM-Kinetics_Kcat_Km_Kcat/Km"].apply(lambda x: [float(i[0]) for i in x])
kcat["CataPro"] = kcat["CataPro_Kcat_Km_Kcat/Km"].apply(lambda x: [float(i[0]) for i in x])
kcat["CatPred"] = kcat["CatPred_Kcat_Km"].apply(lambda x: [float(i[0]) for i in x])
kcat["DLKcat"] = kcat["DLKcat_Kcat"]


km = pd.concat([
                df3.loc[df3["flag"] == "literature_km",sel]
                 ])
km["UniKP"]           = km["UniKP_Kcat_Km_Kcat/Km"].apply(lambda x: [float(i[1]) for i in x])
km["EITLEM-Kinetics"] = km["EITLEM-Kinetics_Kcat_Km_Kcat/Km"].apply(lambda x: [float(i[1]) for i in x])
km["CataPro"]         = km["CataPro_Kcat_Km_Kcat/Km"].apply(lambda x: [float(i[1]) for i in x])
km["CatPred"]         = km["CatPred_Kcat_Km"].apply(lambda x: [float(i[1]) for i in x])


kk = pd.concat([
                df3.loc[df3["flag"].isin(["literature_kk","literature_normalized_fitness","literature_DMS_score"]),sel]
                 ])
kk["UniKP"]           = kk["UniKP_Kcat_Km_Kcat/Km"].apply(lambda x: [float(i[2]) for i in x])
kk["EITLEM-Kinetics"] = kk["EITLEM-Kinetics_Kcat_Km_Kcat/Km"].apply(lambda x: [float(i[2]) for i in x])
kk["CataPro"]         = kk["CataPro_Kcat_Km_Kcat/Km"].apply(lambda x: [float(i[2]) for i in x])

kcat["flag"].value_counts()
km["flag"].value_counts()
kk["flag"].value_counts()


# In[19]:


kcat["n"] = kcat["Value"].apply(lambda x: len(x))
km["n"]   = km["Value"].apply(lambda x: len(x))
kk["n"]   = kk["Value"].apply(lambda x: len(x))


# In[20]:


kcat_corr = kcat.loc[:,["n"]]
km_corr   = km.loc[:,["n"]]
kk_corr   = kk.loc[:,["n"]]

sel = [ 'GEMME', 'VESPA', 'ESM1V', 'SaProt', 'ProSST','UniKP',
       'EITLEM-Kinetics', 'CataPro']

for name in sel:
    kcat_corr[name] = kcat.apply(lambda x: spearmanr(x[name],x["Value"])[0],axis=1)
    km_corr[name]   = km.apply(lambda x: spearmanr(x[name],x["Value"])[0],axis=1)
    kk_corr[name]   = kk.apply(lambda x: spearmanr(x[name],x["Value"])[0],axis=1)
    
kcat_corr["CatPred"] = kcat.apply(lambda x: spearmanr(x["CatPred"],x["Value"])[0],axis=1)
kcat_corr["DLKcat"] = kcat.apply(lambda x: spearmanr(x["DLKcat"],x["Value"])[0],axis=1)

km_corr["CatPred"] = km.apply(lambda x: spearmanr(x["CatPred"],x["Value"])[0],axis=1)

kcat_corr.head(2)
km_corr.head(2)
kk_corr.head(2)


# In[21]:


kcat_corr["VESPA"] = kcat_corr["VESPA"].apply(lambda x: x*(-1))
kk_corr["VESPA"]   = kk_corr["VESPA"].apply(lambda x: x*(-1))

km_corr["GEMME"]  = km_corr["GEMME"].apply(lambda x: x*(-1))
km_corr["ESM1V"]  = km_corr["ESM1V"].apply(lambda x: x*(-1))
km_corr["SaProt"] = km_corr["SaProt"].apply(lambda x: x*(-1))
km_corr["ProSST"] = km_corr["ProSST"].apply(lambda x: x*(-1))

kcat_corr.head(2)
km_corr.head(2)
kk_corr.head(2)


# In[22]:


box_list = [
[ i for i in kcat_corr["GEMME"].tolist()]   + [ i for i in km_corr["GEMME"].tolist()]   + [ i for i in kk_corr["GEMME"].tolist()],
[ i for i in kcat_corr["VESPA"].tolist()]   + [ i for i in km_corr["VESPA"].tolist()]   + [ i for i in kk_corr["VESPA"].tolist()],
[ i for i in kcat_corr["ESM1V"].tolist()]   + [ i for i in km_corr["ESM1V"].tolist()]   + [ i for i in kk_corr["ESM1V"].tolist()],
[ i for i in kcat_corr["SaProt"].tolist()]  + [ i for i in km_corr["SaProt"].tolist()]  + [ i for i in kk_corr["SaProt"].tolist()],
[ i for i in kcat_corr["ProSST"].tolist()]  + [ i for i in km_corr["ProSST"].tolist()]  + [ i for i in kk_corr["ProSST"].tolist()],
[ i for i in kcat_corr["UniKP"].tolist()]           + [ i for i in km_corr["UniKP"].tolist()] + [ i for i in kk_corr["UniKP"].tolist()],
[ i for i in kcat_corr["EITLEM-Kinetics"].tolist()] + [ i for i in km_corr["EITLEM-Kinetics"].tolist()] + [ i for i in kk_corr["EITLEM-Kinetics"].tolist()],
[ i for i in kcat_corr["CataPro"].tolist()]         + [ i for i in km_corr["CataPro"].tolist()] + [i for i in kk_corr["CataPro"].tolist()],
[ i for i in kcat_corr["CatPred"].tolist()]         + [ i for i in km_corr["CatPred"].tolist()] ,
[ i for i in kcat_corr["DLKcat"].tolist()] 
]

sel = ['GEMME', 'VESPA', 'ESM1V', 'SaProt', 'ProSST',          'UniKP', 'EITLEM-Kinetics', 'CataPro', "CatPred","DLKcat"]
col = [ '#c94e65','#c94e65','#c94e65','#c94e65','#c94e65',     '#f47721','#f47721','#f47721','#f47721','#f47721']

cur = pd.DataFrame({"model": sel,"corr":box_list,"col":col})
cur["corr_median"] = cur["corr"].apply(lambda x: np.median(x))
cur = cur.sort_values("corr_median",ascending=False)

box1 = cur.copy()
box1
del cur


# In[23]:


tmp = []
for i in ['GEMME', 'VESPA', 'ESM1V', 'SaProt', 'ProSST',          'UniKP', 'EITLEM-Kinetics', 'CataPro']:
    r_list = kcat_corr[i].tolist() + km_corr[i].tolist() + kk_corr[i].tolist()
    n_list = kcat_corr["n"].tolist() + km_corr["n"].tolist() + kk_corr["n"].tolist()
    tmp.append((i,fisher_z_meta_analysis(r_list, n_list)))
tmp


# In[24]:


r_list = kcat_corr["CatPred"].tolist() + km_corr["CatPred"].tolist()
n_list = kcat_corr["n"].tolist()       + km_corr["n"].tolist()
tmp.append(("CatPred",fisher_z_meta_analysis(r_list, n_list)))

r_list = kcat_corr["DLKcat"].tolist()
n_list = kcat_corr["n"].tolist()
tmp.append(("DLKcat",fisher_z_meta_analysis(r_list, n_list)))

cur = pd.DataFrame({"model": [i[0] for i in tmp],"pooled_r":[i[1] for i in tmp],"col": [ '#c94e65','#c94e65','#c94e65','#c94e65','#c94e65',     '#f47721','#f47721','#f47721','#f47721','#f47721']})
cur = cur.sort_values("pooled_r",ascending=False)

box2 = cur.copy()
box2
del cur


# In[25]:


InteractiveShell.ast_node_interactivity = "last"

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# In[26]:


fig, axes = plt.subplots(1, 2, figsize=(10, 4.7),dpi=300)

plt.subplots_adjust(wspace=0.1, hspace=0.9)

bp = axes[0].boxplot(box1["corr"].tolist(),labels=box1["model"].tolist(),showfliers=False,patch_artist=True)       
axes[0].tick_params(axis='x', rotation=90)
axes[0].set_yticks(np.arange(-1, 1.2, 0.25)) 
axes[0].set_ylim((-1.1,1.1)) 
axes[0].set_ylabel("Spearman Corr",fontsize=12, fontweight='bold')   
axes[0].axhline(y=0, color='black', linestyle='--', linewidth=1)
axes[0].text(-0.18, 1.06, "C", transform=axes[0].transAxes, fontsize=14, fontweight='bold', va='top')

for box,color in zip(bp['boxes'],box1["col"].tolist()):
    box.set(facecolor=color, alpha=0.8)
    
for median in bp['medians']:
    median.set(color='black', linewidth=2)
    
axes[1].bar(box2["model"].tolist(),box2["pooled_r"].tolist(),color=box2["col"].tolist())
axes[1].tick_params(axis='x', rotation=90)   
axes[1].set_yticks(np.arange(-0.3, 0.6, 0.1)) 
axes[1].set_ylabel("Fisher z meta-analysis",fontsize=12, fontweight='bold') 
axes[1].text(-0.18, 1.06, "D", transform=axes[1].transAxes, fontsize=14, fontweight='bold', va='top')

for num in [0,1]:
    # 批量设置x/y轴刻度标签加粗
    for label in axes[num].get_xticklabels():
        label.set_fontweight('bold')
    for label in axes[num].get_yticklabels():
        label.set_fontweight('bold')      
    for spine in axes[num].spines.values():
        spine.set_color('black')    
        spine.set_linewidth(1.5) 
        
legend_elements = [
    Patch(facecolor='#32037d',  label='Stability'),
    Patch(facecolor='#c94e65',  label='Fitness'),
    Patch(facecolor='#46c1be',  label='Binding Affinity'),
    Patch(facecolor='#f47721',  label='Kinetic Parameters')
]
legend = fig.legend(handles=legend_elements, loc='lower center', 
          bbox_to_anchor=(0.5, -0.08), ncol=4, frameon=False, 
          fontsize=12)

for text in legend.get_texts():
    text.set_fontweight('bold')        

plt.tight_layout()
plt.savefig("fig06cd20260415.png", dpi=300, bbox_inches='tight', pad_inches=0.2)
plt.show()


# In[27]:


from PIL import Image

img1 = Image.open("fig06ab20260415.png")
img2 = Image.open("fig06cd20260415.png")

new_img = Image.new(
    "RGB",
    (max(img1.width, img2.width), img1.height + img2.height),
    "white"
)

new_img.paste(img1, (0, 0))
new_img.paste(img2, (0, img1.height))

new_img.save("combined.png")

display(new_img)


# In[ ]:




