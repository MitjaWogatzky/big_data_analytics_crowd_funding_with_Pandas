#!/usr/bin/env python
# coding: utf-8

# In[1]:


from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly_express as px


# # Dateien importieren

# In[3]:


# Datei importieren

df_final = pd.read_csv("df_final.csv", sep=",", index_col=0)
df_dropna_funding = pd.read_csv("df_dropna_funding", sep=",", index_col=0)


# In[4]:


df_final.head()


# In[5]:


df_dropna_funding.head()


# In[6]:


df_final.info()


# In[7]:


df_dropna_funding.info()


# # Data Preprocessing 

# In[8]:


# Datensatz zu groß -> Categorien umwandeln

df_final["activity"] = df_final["activity"].astype("category")
df_final["sector"] = df_final["sector"].astype("category")
df_final["continent"] = df_final["continent"].astype("category")
df_final["country_code"] = df_final["country_code"].astype("category")
df_final["country"] = df_final["country"].astype("category")
df_final["currency"] = df_final["currency"].astype("category")
df_final["sex_majority"] = df_final["sex_majority"].astype("category")
df_final["repayment_interval"] = df_final["repayment_interval"].astype("category")
df_final["team_category"] = df_final["team_category"].astype("category")

df_dropna_funding["activity"] = df_final["activity"].astype("category")
df_dropna_funding["sector"] = df_final["sector"].astype("category")
df_dropna_funding["continent"] = df_final["continent"].astype("category")
df_dropna_funding["country_code"] = df_final["country_code"].astype("category")
df_dropna_funding["country"] = df_final["country"].astype("category")
df_dropna_funding["currency"] = df_final["currency"].astype("category")
df_dropna_funding["sex_majority"] = df_final["sex_majority"].astype("category")
df_dropna_funding["repayment_interval"] = df_final["repayment_interval"].astype("category")
df_dropna_funding["team_category"] = df_final["team_category"].astype("category")

# "Downcasting" von int Datensätzen

df_final[["funded_amount", "loan_amount", "term_in_months", "lender_count"]] = df_final[["funded_amount", "loan_amount", "term_in_months", "lender_count"]].apply(pd.to_numeric, downcast="unsigned")
df_dropna_funding[["funded_amount", "loan_amount", "term_in_months", "lender_count"]] = df_final[["funded_amount", "loan_amount", "term_in_months", "lender_count"]].apply(pd.to_numeric, downcast="unsigned")


# In[9]:


df_final.info()


# In[10]:


df_dropna_funding.info()


# # Pairplot metrischen Variablen

# In[11]:


sns.pairplot(data=df_dropna_funding, corner=True) 


# ## Erkenntnisse Pairplot
# 
# Grundlegegend ist zu beachten, dass aufgrund der hohen Menge an Finanzierungsprojekten (671.204 Projekte) und somit ebenso großen Anzahl an Punkten je Graphik, keine Aussagen bzgl. Mengenverteilungen getroffen werden können. Dafür ist die Puntkewolke zu dicht. 
# 
# + __funded_amount__:Der Großteil der finanzierten Projekte hat ein Volumen bis zu 10.000 US-Dollar. Es gab keine Finanzierung, die höher lag, als der gewünschte Betrag; Augenscheinlich gibt es einen Zusammenhang - je höhe die Finzierungssumme, desto höher die Anzahl der Darlehendsgeber
# + __success_ratio__: Der Löwenanteil der Projekte hat 100% der gewünschten Summe als Funding erhalten.
# + __term_in_month__: Der Großteil der Projekte hat eine Finanzierungsdauer unter 50 Monaten
# + __team_count__: Der Großteil der Projekte hat eine sehr kleine Teamgröße (ein bis zwei Personen); je größer die Teams, desto höher die Wahrscheinlichkeit, dass mehr Männer ein einem Team sind, als Frauen

# # Sektoren

# ## Anzahl Projekte

# In[12]:


# nach Anzahl Projekte

df_sector = df_final.groupby("sector").agg({"sector":np.size, "funded_amount":np.sum})
df_sector_rename = df_sector.rename(columns={"sector": "amount_projects"})
df_sector = df_sector_rename.reset_index()
df_sector_sorted = df_sector.sort_values(by=["amount_projects"])
#df_sector_sorted_fund = df_sector.sort_values(by=["funded_amount"])

sns.set(rc={'figure.figsize':(12,9)})
sns.set_theme(style="whitegrid")
ax = sns.barplot(x="sector", y="amount_projects", data=df_sector_sorted, palette="Blues", order=df_sector_sorted["sector"])
for item in ax.get_xticklabels(): item.set_rotation(45)
for i, v in enumerate(df_sector_sorted["amount_projects"].iteritems()):        
    ax.text(i ,v[1], "{:,}".format(v[1]), color='darkred', va ='bottom', rotation=45)
plt.tight_layout()
plt.show()


# ## Fundingdurchschnitt und Projektanzahl

# In[13]:


# Spalten erstellen
df_sector = df_final.groupby("sector").agg({"sector":np.size, "funded_amount":np.sum})
funded_sum = df_sector["funded_amount"].tolist()
df_sector = df_final.groupby("sector").agg({"sector":np.size, "funded_amount":np.mean})
funded_mean = df_sector["funded_amount"].tolist()
df_sector = df_final.groupby("sector").agg({"sector":np.size, "funded_amount":np.median})
funded_median = df_sector["funded_amount"].tolist()

# DataFrame erstellen und Spalten einfügen
df_sector = df_final.groupby("sector").agg({"sector":np.size})
df_sector = df_sector.rename(columns={"sector": "amount_projects"})
df_sector_fund = df_sector.reset_index()
df_sector_fund.insert(2,'funded_sum',funded_sum)
df_sector_fund.insert(3,'funded_mean',funded_mean)
df_sector_fund.insert(4,'funded_median',funded_median)
df_sector_fund

# Plotten
fig = px.scatter(df_sector_fund, x="funded_mean", y="amount_projects", color="sector", size="funded_sum", 
           hover_name="sector", size_max=60)
fig.show()


# ## Funding-Höhe und Funding Erfolg

# In[14]:


# nach Anzahl Projekten

# Nach Sektor und Mittelwerten gruppieren
df_group_sector_mean = df_final.groupby("sector").agg({"sector":np.size, "funded_amount":np.mean, "loan_amount":np.mean, "success_ratio":np.mean, "team_count":np.mean, "term_in_months":np.mean, "lender_count":np.mean})
df_group_sector_rename = df_group_sector_mean.rename(columns={"sector": "amount_projects"})
df_group_sector_mean = df_group_sector_rename.reset_index()
# Plotten
fig = px.scatter(df_group_sector_mean, x="loan_amount", y="funded_amount", size="amount_projects", color="sector",
           hover_name="sector", size_max=60)
fig.show()


# **ERKENNTNISSE**
# 
# **Projektanzahl:**
# Die **meisten Projete** mit 180.301 Projekten hat der Agrarbereich, gefolgt von "Nahrung" mit 136.657 und Einzelhandel mit 124.494 Projekten.
# Die **wenigsten Projekte** gibt es im Bereich "Großhandel" mit lediglich 634 Projekten, gefolgt von "Entertainment" mit 830 Projekten und Manufactoring mit bereits 6.208 Projekten.
# 
# **Fundingvolumen:**
# Der  Funding-Niveau ist ingesamt sehr niedirg. Der Mittelwert aller erhaltenen Fundingvolumen reicht von 411 US-Dollar im Sektor "Personal Use" bis 1570 bei "Wholesale". Der Großteil befindet sich zwischen 600 und 800 US Dollar.
# 
# **Funding-Erfolg**
# Der lineare Verlauf zeigt, dass die Höhe der ausgezahlten Fundingsumme in allen Sektoren nahezu der gewünschten Höhe entspricht (Funding-Erflolg: 94% "Housing bis 99% "Manufacturing"). Einzig Entertainment liegt augenscheinlich unter diesem Bild (Funding-Erfolg: 89%). 
# 
# **Legende:**
# + loan_amount: Mittelwerte gewünschte Fundingsumme in US-Dollar
# + funded_amount: Mittelwerte erhaltene Fundingsumme in US-Dollar
# + Projektanzahl (sector_size): Anzahl Projekte je Sektor 
# + Funding-Erfolg: Das Verhältnis von gewünschter Funding-Summe und erhaltener Fundingsumme

# ## Aktivitäten je Sektor

# In[15]:


df_activity = df_final.groupby(["sector", "activity"]).agg({"activity":np.size, "funded_amount":np.mean})
df_activity = df_activity.dropna()
funded_mean = df_activity["funded_amount"].tolist()
df_activity = df_final.groupby(["sector", "activity"]).agg({"activity":np.size, "funded_amount":np.median})
df_activity = df_activity.dropna()
funded_median = df_activity["funded_amount"].tolist()

df_activity = df_final.groupby(["sector", "activity"]).agg({"activity":np.size, "funded_amount":np.sum})
df_activ = df_activity.rename(columns={"activity": "amount_projects"})
df_activity = df_activ.reset_index()
df_activity = df_activity.loc[df_activity["funded_amount"]!=0, :]
df_activity.insert(4,'funded_mean',funded_mean)
df_activity.insert(5,'funded_median',funded_median)
df_activity = df_activity.sort_values(by=["funded_mean"])
df_activity = df_activity.loc[df_activity["funded_mean"]< 1900, :]
#df_activity = df_activity.loc[df_activity["activity"]!="Renewable Energy Products", :]
#df_activity = df_activity.loc[df_activity["activity"]!="Landscaping / Gardening", :]

fig = px.treemap(df_activity, path=[px.Constant('Overall fundings'), 'sector', 'activity'], values='amount_projects',
                  color='funded_mean', hover_name="activity")
fig.show()


# In[16]:


df_activity = df_activity.sort_values(by=["funded_mean"])
df_activity.head(10)


# In[17]:


df_activity = df_activity.sort_values(by=["funded_mean"])
df_activity.tail(10)


# 

# # Kontinente und Länder

# ## Anzahl Projekte nach Kontinenten

# In[18]:


# Daten vorbereiten
df_cont_count = df_final.groupby(["continent", "country"]).agg({"continent":np.size, "funded_amount":np.sum})
df_cont_count_rename = df_cont_count.rename(columns={"continent": "amount_projects"})
df_cont_count = df_cont_count_rename.reset_index()
df_cont_count_sorted = df_cont_count.sort_values(by=["amount_projects"])
df_cont_count_sorted_fund = df_cont_count.sort_values(by=["funded_amount"])

# Plotten
fig = px.bar(df_cont_count_sorted_fund, x="continent", y="amount_projects", hover_name="country")
fig.show()


# ## Fundingdurchschnitt und Projektanzahl

# ### Alle Länder

# In[19]:


df_cont_count = df_final.groupby(["continent", "country"]).agg({"continent":np.size, "funded_amount":np.sum})
funded_amount = df_cont_count["funded_amount"].tolist()
df_cont_count = df_final.groupby(["continent", "country"]).agg({"continent":np.size, "funded_amount":np.mean})
country_mean = df_cont_count["funded_amount"].tolist()
df_cont_count = df_final.groupby(["continent", "country"]).agg({"continent":np.size, "funded_amount":np.median})
country_median = df_cont_count["funded_amount"].tolist()

df_cont_count = df_final.groupby(["continent", "country"]).agg({"country":np.size})
df_cont_count_rename = df_cont_count.rename(columns={"country": "amount_projects"})
df_cont_count = df_cont_count_rename.reset_index()
df_cont_count.insert(3,'funded_amount',funded_amount)
df_cont_count.insert(4,'country_mean',country_mean)
df_cont_count.insert(5,'country_median',country_median)
df_cont_count = df_cont_count.sort_values(by=["country_mean"])
df_cont_count = df_cont_count.loc[df_cont_count["funded_amount"]!=0, :]
#df_cont_count = df_cont_count.loc[df_cont_count["country_mean"]<50000, :]
df_cont_count


# In[20]:


df_cont_count = df_final.groupby(["continent", "country"]).agg({"continent":np.size, "funded_amount":np.sum})
funded_amount = df_cont_count["funded_amount"].tolist()
df_cont_count = df_final.groupby(["continent", "country"]).agg({"continent":np.size, "funded_amount":np.mean})
country_mean = df_cont_count["funded_amount"].tolist()
df_cont_count = df_final.groupby(["continent", "country"]).agg({"continent":np.size, "funded_amount":np.median})
country_median = df_cont_count["funded_amount"].tolist()

df_cont_count = df_final.groupby(["continent", "country"]).agg({"country":np.size})
df_cont_count_rename = df_cont_count.rename(columns={"country": "amount_projects"})
df_cont_count = df_cont_count_rename.reset_index()
df_cont_count.insert(3,'funded_amount',funded_amount)
df_cont_count.insert(4,'country_mean',country_mean)
df_cont_count.insert(5,'country_median',country_median)
df_cont_count = df_cont_count.sort_values(by=["funded_amount"])
df_cont_count = df_cont_count.loc[df_cont_count["funded_amount"]!=0, :]
df_cont_count = df_cont_count.loc[df_cont_count["country_mean"]<15000, :]
df_cont_count

fig = px.scatter(df_cont_count, x="country_mean", y="amount_projects", color="continent", size="funded_amount", 
           hover_name="country", log_x=True, size_max=60)
fig.show()


# ### Nach Kontinenten

# In[21]:


# Logarithmierte Darstellung

fig = px.scatter(df_cont_count, x="country_mean", y="amount_projects",
           color="continent", hover_name="country", log_x=True, facet_col="continent", size="funded_amount")
fig.show()


# ## Anzahl Projekt vs Fundingsumme je Land und Kontinent

# In[22]:


fig = px.scatter(df_cont_count, x="amount_projects", y="funded_amount", color="continent",
           hover_name="country", log_x=True, size_max=60)
fig.show()


# In[23]:


# Logarithmierte Darstellung

fig = px.scatter(df_cont_count, x="amount_projects", y="funded_amount",
           color="continent", hover_name="country", log_x=True, facet_col="continent")
fig.show()


# # Länder und Sektoren im Verhältnis

# In[24]:


df_activity


# In[25]:


df_count_sect = df_final.groupby(["continent", "country", "sector", "activity"]).agg({"funded_amount":np.sum, "activity":np.size})
df_activ = df_count_sect.rename(columns={"activity": "amount_projects"})
df_count_sect = df_activ.reset_index()


# In[26]:


df_count_sect = df_count_sect.dropna()
df_count_sect = df_count_sect.sort_values(by=["funded_amount"])
df_count_sect = df_count_sect.loc[df_count_sect["funded_amount"]< 6000000, :]
df_count_sect.tail(10)


# In[27]:


fig = px.treemap(df_count_sect, path=[px.Constant('world'), 'continent', 'country','sector'], values='amount_projects',
                  color='funded_amount')
fig.show()


# In[82]:


df_iso = df_final.groupby(["ISO_ALPHA3", "sector", "activity"]).agg({"funded_amount":np.sum, "activity":np.size})
df_activ = df_iso.rename(columns={"activity": "amount_projects"})
df_iso = df_activ.reset_index()


# In[56]:


df_iso = df_iso.dropna()
df_iso.loc[df_iso["funded_amount"]==0, :]


# In[84]:


df_iso = df_iso.dropna()
df_iso.loc[df_iso["funded_amount"]==0, :]
df_iso = df_iso.sort_values(by=["funded_amount"])
df_iso["amount_projects"] = df_iso["amount_projects"].astype("int")
#df_iso["amount_projects"] = df_iso["funded_amount"].astype("int")
#df_iso = df_iso.loc[df_iso["funded_amount"]< 100000, :]
df_iso


# In[81]:


fig = px.choropleth(df_iso, locations="ISO_ALPHA3",
                    color="amount_projects", # lifeExp is a column of gapminder
                    #hover_name="country", # column to add to hover information
                    color_continuous_scale=px.colors.sequential.Plasma)
fig.show()


# In[30]:


fig = px.choropleth(df_iso, locations="ISO_ALPHA3", color="amount_projects", animation_frame="sector")
fig.show()


# # Beeinflusst die Darlehenshöhe die Anzahl der Darlehensgeber?

# In[31]:


# Daten für ax1 vorbereiten

df_funding_sector = df_final.groupby("sector").agg({"funded_amount":np.mean})
df_fund_sector_sorted = df_funding_sector.sort_values(by=["funded_amount"], ascending=True)
df_fund_sector_sorted

# Daten für ax2 vorbereiten

df_funding_lender = df_final.groupby("sector").agg({"lender_count":np.mean,"funded_amount":np.mean})
df_funding_lender_sorted = df_funding_lender.sort_values(by=["funded_amount"], ascending=True)
df_funding_lender_sorted

# Plot erstellen
 
fig, ax1 = plt.subplots(figsize=(10,10))
sns.set_style("whitegrid")
sns.barplot(x=df_fund_sector_sorted.index, y=df_fund_sector_sorted["funded_amount"], palette="Blues",data=df_fund_sector_sorted, ax=ax1)

ax1.xaxis.set_tick_params(rotation=70, labelsize=10)
ax1.set_ylabel("Durchschnitt Darlehenshöhe in US-Dollar (Balken)", fontsize=14)
ax1.set_xlabel("Sektor", fontsize=14)

ax2 = ax1.twinx()
ax2 = sns.lineplot(x=df_funding_lender_sorted.index, y=df_funding_lender_sorted["lender_count"], data=df_funding_lender_sorted, color="darkred")
ax2.set_ylabel('Durchschnitt Anzahl Darlehensgeber (Linie)', fontsize=14)

plt.show()


# # Überzeugen weniger erfolgreiche Teams auch weniger die Investoren?

# In[32]:


# Daten vorbereiten

# Elfenbeinküste, Mauretanien, Buthan und Afghanistan entfernen, da sie bis zu 17 mal so viele Darlehensgeber hat, 
# wie die anderen 95% der Länder und somit die Graphik stark verzerrt. Virgin Islands entfernen, da keine 
# Projekte gefördert wurden. Die Entferrnungen haben keinen Einfluss auf die Aussage der Graphik.

df_final = df_final.loc[df_final["country_code"]!="CI",:]
df_final = df_final.loc[df_final["country_code"]!="MR",:]
df_final = df_final.loc[df_final["country_code"]!="BT",:]
df_final = df_final.loc[df_final["country_code"]!="AF",:]
df_final = df_final.loc[df_final["country_code"]!="VI",:]

# DataFrame mit erfolgreichen Fundings (Erfolgsquote = 100%), mit wenig erfolgreichen (Erfolgsquote < 50%) 
# und dazwischen liegenden erstellen

df_success = df_final.loc[df_final["success_classes"]=="Gleich100",:]
df_no_success = df_final.loc[df_final["success_classes"]=="KleinerGleich50",:]


# Daten gruppieren

# Daten für ax1
df_fund_country = df_final.groupby("country").agg({"lender_count":np.mean, "funded_amount":np.mean})
df_fund_country_sorted = df_fund_country.sort_values(by=["funded_amount"], ascending=True)

# Daten für ax2_success 
df_succ_sect = df_success.groupby("country").agg({"lender_count":np.mean, "funded_amount":np.mean})
df_succ_sect_sorted = df_succ_sect.sort_values(by=["funded_amount"], ascending=True)

# Daten für ax3_no_success 
df_no_succ_sect = df_no_success.groupby("country").agg({"lender_count":np.mean, "funded_amount":np.mean})
df_no_succ_sect_sorted = df_no_succ_sect.sort_values(by=["funded_amount"], ascending=True)


# Daten plotten

fig, ax1 = plt.subplots(figsize=(14,14))
sns.set_style("whitegrid")
sns.barplot(x=df_fund_country_sorted.index, y=df_fund_country_sorted["lender_count"], color="grey",data=df_fund_country_sorted, ax=ax1)

ax1.xaxis.set_tick_params(rotation=90, labelsize=9)
ax1.set_ylabel("Median Anzahl Darlehensgeber, alle Projekte (graue Balken)", fontsize=14)
ax1.set_xlabel("Länder", fontsize=14)

ax2 = sns.lineplot(x=df_succ_sect_sorted.index, y=df_succ_sect_sorted["lender_count"], data=df_succ_sect_sorted, color="lightcoral")
ax3 = sns.lineplot(x=df_no_succ_sect_sorted.index, y=df_no_succ_sect_sorted["lender_count"], data=df_no_succ_sect_sorted, color="darkred")

ax2.set_ylabel('Durchschnitt Anzahl Darlehensgeber: alle (Balken), erfolgreiche (rosa), nicht erfolgreiche (rot) Projekte', fontsize=14)

plt.show()


# # Ausgewählte Zahlen

# In[33]:


# Übersicht bzgl. der Verteilungen
df_final.describe()


# In[34]:


# sector_size: absolute Anzahl Projekte
# Alle weiteren Variablen: MITTELWERT

df_group_activity = df_final.groupby("activity").agg({"sector":np.size, "funded_amount":np.mean, "loan_amount":np.mean, "success_ratio":np.mean, "team_count":np.mean, "term_in_months":np.mean, "lender_count":np.mean})
df_group_sector_rename = df_group_activity.rename(columns={"sector": "activity_size"})
df_group_activity = df_group_sector_rename.reset_index()
df_group_activity


# In[35]:


# sector_size: absolute Anzahl Projekte
# Alle weiteren Variablen: MEDIAN

df_group_sector_median = df_final.groupby("sector").agg({"sector":np.size, "funded_amount":np.median, "loan_amount":np.median, "success_ratio":np.median, "team_count":np.median, "term_in_months":np.median, "lender_count":np.median})
df_group_sector_rename = df_group_sector_median.rename(columns={"sector": "sector_size"})
df_group_sector_median = df_group_sector_rename.reset_index()
df_group_sector_median


# # Kontinente

# In[36]:


df_cont = df_dropna_funding.groupby("continent").agg({"continent":np.size, "funded_amount":np.sum})
df_cont_rename = df_cont.rename(columns={"continent": "amount_projects"})
df_cont = df_cont_rename.reset_index()
df_cont_sorted = df_cont.sort_values(by=["amount_projects"])
df_cont_sorted_fund = df_cont.sort_values(by=["funded_amount"])
df_cont_sorted


# In[37]:


fig = px.scatter(df_cont_count, x="amount_projects", y="funded_amount", color="continent",
           hover_name="country", log_x=True, size_max=60)
fig.show()


# In[38]:


fig = px.bar(df_cont_count_sorted, x="amount_projects", y="country", barmode="group")
fig.update_layout(
    font_size = 8,
    autosize=False,
    width=900,
    height=1000,
    margin=dict(
        l=50,
        r=50,
        b=100,
        t=100,
        pad=4
    ),
    paper_bgcolor="LightSteelBlue",
)
fig.show()


# In[39]:


fig = px.pie(df_cont, values='amount_projects', names=['Africa', 'Asia', 'Europe', 'North America', "Oceania", "South America"], title='payment interval')
fig.show()


# In[40]:


fig = px.pie(df_cont, values='funded_amount', names=['Africa', 'Asia', 'Europe', 'North America', "Oceania", "South America"], title='payment interval')
fig.show()


# In[41]:


fig = px.scatter(df_group_sector_mean, x="lender_count", y="funded_amount", size="sector_size", color="team_count",
           hover_name="sector", size_max=60)
fig.show()


# ##### ERKENNTNISSE
# 
# Es besteht augenscheinlich ein Zusammenhang zwischen Anzahl Darlehnensgeber und ausgezahltem Fundingbetrag. Je höher der Betrag, desto mehr 
# Kein Zusammenhang zwischen Teamgröße ausgezahltem Fundingbetrag, als auch Anzahl Darlehensgeber

# In[ ]:


# Daten zur Graphik - MITTELWERTE
# sector_size: absolute Anzahl Projekte
# Alle weiteren Variablen: Mittelwert

df_group_sector_mean


# In[ ]:


# Daten wie oben, jedoch zum Vergleich mit MEDIANEN
# sector_size: absolute Anzahl Projekte
# Alle weiteren Variablen: Median

df_group_sector_median = df_final.groupby("sector").agg({"sector":np.size, "funded_amount":np.median, "loan_amount":np.median, "success_ratio":np.median, "team_count":np.median, "term_in_months":np.median, "lender_count":np.median})
df_group_sector_rename = df_group_sector_median.rename(columns={"sector": "sector_size"})
df_group_sector_median = df_group_sector_rename.reset_index()
df_group_sector_median


# In[ ]:





# In[ ]:


fig = px.scatter(df_group_sector_mean, x="loan_amount", y="funded_amount", size="sector_size", color="lender_count",
           hover_name="sector", size_max=60)
fig.show()


# In[ ]:


fig = px.scatter(df_group_sector_mean, x="team_count", y="funded_amount", size="sector_size", color="lender_count",
           hover_name="sector", size_max=60)
fig.show()


# In[ ]:


fig = px.scatter(df_group_sector_mean, x="lender_count", y="funded_amount", size="sector_size", color="team_count",
           hover_name="sector", size_max=60)
fig.show()


# In[ ]:


# Nach Aktivität groupieren gruppieren

df_group_activity = df_final.groupby("activity").agg({"sector":np.size, "funded_amount":np.mean, "loan_amount":np.mean, "success_ratio":np.mean, "team_count":np.mean, "term_in_months":np.mean, "lender_count":np.mean})
df_group_sector_rename = df_group_activity.rename(columns={"sector": "activity_size"})
df_group_activity = df_group_sector_rename.reset_index()
df_group_activity


# In[ ]:


fig = px.scatter(df_group_activity, x="funded_amount", y="lender_count", color="term_in_months",
           hover_name="activity", size_max=70)
fig.show()


# ## Länder bzgl. Projektanzahl, Funding-Höhe und Funding Erfolg

# In[ ]:


df_final.info()


# In[ ]:


# Nach Aktivität groupieren gruppieren

df_group_activity = df_final.groupby("activity").agg({"sector":np.size, "funded_amount":np.mean, "loan_amount":np.mean, "success_ratio":np.mean, "team_count":np.mean, "term_in_months":np.mean, "lender_count":np.mean})
df_group_activity


# In[ ]:


df_final.loc[df_final["activity"]=="Adult Care", :]


# In[ ]:


# Nach Aktivität groupieren gruppieren

df_group_activity = df_final.groupby(["sector", "activity"]).agg({"sector":np.size, "funded_amount":np.mean, "loan_amount":np.mean, "success_ratio":np.mean, "team_count":np.mean, "term_in_months":np.mean, "lender_count":np.mean})
df_group_activity


# In[ ]:





# In[ ]:


fig = px.scatter(df_group_sector, x="loan_amount", y="funded_amount", size="sector_size", color="lender_count",
           hover_name="sector", size_max=60)
fig.show()


# 

# ## Länder bzgl. Projektanzahl, Funding-Höhe und Funding Erfolg

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Verteilung Länder und Sektoren 

# In[ ]:


fig = px.treemap(df_dropna_funding, path=[px.Constant('world'), 'continent', 'country', "sector"], values='funded_amount',
                  color='funded_amount')
fig.show()


# # Drei Haupt-Visualisierungen 

# ## Beeinflusst der Sektor die Anzahl Darlehensgeber und -höhe?

# In[ ]:


# Daten für ax1 vorbereiten

df_funding_sector = df_final.groupby("sector").agg({"funded_amount":np.mean})
df_fund_sector_sorted = df_funding_sector.sort_values(by=["funded_amount"], ascending=True)
df_fund_sector_sorted

# Daten für ax2 vorbereiten

df_funding_lender = df_final.groupby("sector").agg({"lender_count":np.mean,"funded_amount":np.mean})
df_funding_lender_sorted = df_funding_lender.sort_values(by=["funded_amount"], ascending=True)
df_funding_lender_sorted

# Plot erstellen
 
fig, ax1 = plt.subplots(figsize=(10,10))
sns.set_style("whitegrid")
sns.barplot(x=df_fund_sector_sorted.index, y=df_fund_sector_sorted["funded_amount"], palette="Blues",data=df_fund_sector_sorted, ax=ax1)

ax1.xaxis.set_tick_params(rotation=70, labelsize=10)
ax1.set_ylabel("Median Höhe Darlehen", fontsize=14)
ax1.set_xlabel("Sektor", fontsize=14)

ax2 = ax1.twinx()
ax2 = sns.lineplot(x=df_funding_lender_sorted.index, y=df_funding_lender_sorted["lender_count"], data=df_funding_lender_sorted, color="darkred")
ax2.set_ylabel('Median Anzahl Darlehensgeber', fontsize=14)

plt.show()


# In[ ]:


# Daten für ax1 vorbereiten

df_funding_sector = df_final.groupby("sector").agg({"funded_amount":np.median})
df_fund_sector_sorted = df_funding_sector.sort_values(by=["funded_amount"], ascending=True)
df_fund_sector_sorted


# In[ ]:


# Daten für ax2 vorbereiten

df_funding_lender = df_final.groupby("sector").agg({"lender_count":np.median,"funded_amount":np.median})
df_funding_lender_sorted = df_funding_lender.sort_values(by=["funded_amount"], ascending=True)
df_funding_lender_sorted


# In[ ]:





# In[ ]:


# Plot erstellen
 
fig, ax1 = plt.subplots(figsize=(10,10))
sns.set_style("whitegrid")
sns.barplot(x=df_fund_sector_sorted.index, y=df_fund_sector_sorted["funded_amount"], palette="Blues",data=df_fund_sector_sorted, ax=ax1)

ax1.xaxis.set_tick_params(rotation=70, labelsize=10)
ax1.set_ylabel("Median Höhe Darlehen", fontsize=14)
ax1.set_xlabel("Sektor", fontsize=14)

ax2 = ax1.twinx()
ax2 = sns.lineplot(x=df_funding_lender_sorted.index, y=df_funding_lender_sorted["lender_count"], data=df_funding_lender_sorted, color="darkred")
ax2.set_ylabel('Median Anzahl Darlehensgeber', fontsize=14)

plt.show()


# ##### Fragestellung bei Erstellung der Graphik: 
# 
# Beeinflusst der Sektor die Anzahl Darlehensgeber und -höhe?
# 
# Aus dem Data Preprocessing Pairplot ist bereits bekant, dass Anzahl Darlehensgeber steigt, je höher das Darlehenen ist.
# 
# Nun interessiert, ob ...
# - dieser Zusammenhang je Sektor unterschiedlich ist. 
# - Müssen also Gründer je Sektor gegebenfalls weniger oder mehr Darlehensgeber überzeugen, um die für den Sektor übliche erforderliche Summe zu erhalten?  
#  
# 
# ##### Erkenntnisse
# 
# - Dieser Zusammenhang besteht auch je Sektor überwiegend weiterhin.
# 
# - Einzig bei "Personal Use" braucht es im Verhältnis zur Darlehenshöhe sichtbar weniger Darlehensgeber sowie bei "Wholesale" im Verhältnis mehr. Dies könnte sich aus dem insgesamt niedrigen Darlehensnivau bei "Personal Use" bzw. bei dem hohen von "Wholesale" erklären. Für einen geringe Darlehenshöhe werden auch weniger Darlehensgeber benötigt.
# 

# ## Überzeugen weniger erfolgreiche Teams auch weniger die Investoren? 

# In[ ]:


df_success = df_final.loc[df_final["success_ratio"]==100,:]
df_success_medium = df_final.loc[(df_final["success_ratio"]<100) & (df_final["success_ratio"]>50)]
df_no_success = df_final.loc[(df_final["success_ratio"]<=50),:]


# In[ ]:


df_success = df_final.loc[df_final["success_ratio"]==100,:]
df_success


# In[ ]:


df_success_medium = df_final.loc[(df_final["success_ratio"]<100) & (df_final["success_ratio"]>50)]
df_success_medium


# In[ ]:


# DataFrame mit wenig erfolgreichen Fundings erstellen (Erfolgsquote kleiner gleich fünfzig)

df_no_success = df_final.loc[(df_final["success_ratio"]<=50),:]
df_no_success


# In[ ]:


# DataFrame mit erfolgreichen Fundings (Erfolgsquote = 100%), mit wenig erfolgreichen (Erfolgsquote < 50%) 
# und dazwischen liegenden erstellen

df_success = df_final.loc[df_final["success_classes"]=="Gleich100",:]
df_success_medium = df_final.loc[df_final["success_classes"]=="Größer50Kleiner100",:]
df_no_success = df_final.loc[df_final["success_classes"]=="KleinerGleich50",:]

# Elfenbeinküste, Mauretanien, Buthan und Afghanistan entfernen, da sie bis zu 17 mal so viele Darlehensgeber hat, 
# wie die anderen 95% der Länder und somit die Graphik stark verzerrt. Virgin Islands entfernen, da keine 
# Projekte gefördert wurden. Die Entferrnungen haben keinen Einfluss auf die Aussage der Graphik.

df_final = df_final.loc[df_final["country_code"]!="CI",:]
df_final = df_final.loc[df_final["country_code"]!="MR",:]
df_final = df_final.loc[df_final["country_code"]!="BT",:]
df_final = df_final.loc[df_final["country_code"]!="AF",:]
df_final = df_final.loc[df_final["country_code"]!="VI",:]

df_success = df_success.loc[df_success["country_code"]!="CI",:]
df_success = df_success.loc[df_success["country_code"]!="MR",:]
df_success = df_success.loc[df_success["country_code"]!="BT",:]
df_success = df_success.loc[df_success["country_code"]!="AF",:]
df_success = df_success.loc[df_success["country_code"]!="VI",:]

df_success_medium = df_success_medium.loc[df_success_medium["country_code"]!="CI",:]
df_success_medium = df_success_medium.loc[df_success_medium["country_code"]!="MR",:]
df_success_medium = df_success_medium.loc[df_success_medium["country_code"]!="BT",:]
df_success_medium = df_success_medium.loc[df_success_medium["country_code"]!="AF",:]
df_success_medium = df_success_medium.loc[df_success_medium["country_code"]!="VI",:]

df_no_success = df_no_success.loc[df_no_success["country_code"]!="CI",:]
df_no_success = df_no_success.loc[df_no_success["country_code"]!="MR",:]
df_no_success = df_no_success.loc[df_no_success["country_code"]!="BT",:]
df_no_success = df_no_success.loc[df_no_success["country_code"]!="AF",:]
df_no_success = df_no_success.loc[df_no_success["country_code"]!="VI",:]


# In[ ]:


# Daten für ax1 vorbereiten
df_fund_country = df_final.groupby("country").agg({"lender_count":np.mean, "funded_amount":np.mean})
df_fund_country_sorted = df_fund_country.sort_values(by=["funded_amount"], ascending=True)

# Daten für ax2_success vorbereiten
df_succ_sect = df_success.groupby("country").agg({"lender_count":np.mean, "funded_amount":np.mean})
df_succ_sect_sorted = df_succ_sect.sort_values(by=["funded_amount"], ascending=True)

# Daten für ax3_no_success vorbereiten
df_no_succ_sect = df_no_success.groupby("country").agg({"lender_count":np.mean, "funded_amount":np.mean})
df_no_succ_sect_sorted = df_no_succ_sect.sort_values(by=["funded_amount"], ascending=True)


# In[ ]:


df_final = df_final.loc[df_final["country_code"]!=("CI", "MR", "BT", "AF", "VI"),:]
df_success = df_success.loc[df_success["country_code"]!=("CI", "MR", "BT", "AF", "VI"),:]
df_no_success = df_no_success.loc[df_no_success["country_code"]!=("CI", "MR", "BT", "AF", "VI"),:]

671204


# In[ ]:


df_final


# In[ ]:


df_test = df_final.copy()


# In[ ]:


indexNames = df_test[(df_test['country_code'] == "CI") | (df_test['country_code'] == "MR") | 
                      (df_test['country_code'] == "BT") | (df_test['country_code'] == "AF") |
                     (df_test['country_code'] == "VI")].index 
df_test.drop(indexNames, inplace=True)


# In[ ]:


df_test


# In[ ]:



df_final = df_final.loc[(df_final["country_code"]!="CI",:]
df_final = df_final.loc[df_final["country_code"]!="MR",:]
df_final = df_final.loc[df_final["country_code"]!="BT",:]
df_final = df_final.loc[df_final["country_code"]!="AF",:]
df_final = df_final.loc[df_final["country_code"]!="VI",:]

df_success = df_success.loc[df_success["country_code"]!="CI",:]
df_success = df_success.loc[df_success["country_code"]!="MR",:]
df_success = df_success.loc[df_success["country_code"]!="BT",:]
df_success = df_success.loc[df_success["country_code"]!="AF",:]
df_success = df_success.loc[df_success["country_code"]!="VI",:]

df_no_success = df_no_success.loc[df_no_success["country_code"]!="CI",:]
df_no_success = df_no_success.loc[df_no_success["country_code"]!="MR",:]
df_no_success = df_no_success.loc[df_no_success["country_code"]!="BT",:]
df_no_success = df_no_success.loc[df_no_success["country_code"]!="AF",:]
df_no_success = df_no_success.loc[df_no_success["country_code"]!="VI",:]


# In[ ]:


# Daten vorbereiten
# DataFrame mit erfolgreichen Fundings (Erfolgsquote = 100%), mit wenig erfolgreichen (Erfolgsquote < 50%) 
# und dazwischen liegenden erstellen

df_success = df_final.loc[df_final["success_classes"]=="Gleich100",:]
df_no_success = df_final.loc[df_final["success_classes"]=="KleinerGleich50",:]

# Elfenbeinküste, Mauretanien, Buthan und Afghanistan entfernen, da sie bis zu 17 mal so viele Darlehensgeber hat, 
# wie die anderen 95% der Länder und somit die Graphik stark verzerrt. Virgin Islands entfernen, da keine 
# Projekte gefördert wurden. Die Entferrnungen haben keinen Einfluss auf die Aussage der Graphik.

df_final = df_final.loc[df_final["country_code"]!="CI",:]
df_final = df_final.loc[df_final["country_code"]!="MR",:]
df_final = df_final.loc[df_final["country_code"]!="BT",:]
df_final = df_final.loc[df_final["country_code"]!="AF",:]
df_final = df_final.loc[df_final["country_code"]!="VI",:]

df_success = df_success.loc[df_success["country_code"]!="CI",:]
df_success = df_success.loc[df_success["country_code"]!="MR",:]
df_success = df_success.loc[df_success["country_code"]!="BT",:]
df_success = df_success.loc[df_success["country_code"]!="AF",:]
df_success = df_success.loc[df_success["country_code"]!="VI",:]

df_no_success = df_no_success.loc[df_no_success["country_code"]!="CI",:]
df_no_success = df_no_success.loc[df_no_success["country_code"]!="MR",:]
df_no_success = df_no_success.loc[df_no_success["country_code"]!="BT",:]
df_no_success = df_no_success.loc[df_no_success["country_code"]!="AF",:]
df_no_success = df_no_success.loc[df_no_success["country_code"]!="VI",:]

# Daten gruppieren
# Daten für ax1
df_fund_country = df_final.groupby("country").agg({"lender_count":np.mean, "funded_amount":np.mean})
df_fund_country_sorted = df_fund_country.sort_values(by=["funded_amount"], ascending=True)

# Daten für ax2_success 
df_succ_sect = df_success.groupby("country").agg({"lender_count":np.mean, "funded_amount":np.mean})
df_succ_sect_sorted = df_succ_sect.sort_values(by=["funded_amount"], ascending=True)

# Daten für ax3_no_success 
df_no_succ_sect = df_no_success.groupby("country").agg({"lender_count":np.mean, "funded_amount":np.mean})
df_no_succ_sect_sorted = df_no_succ_sect.sort_values(by=["funded_amount"], ascending=True)


# Plotten
fig, ax1 = plt.subplots(figsize=(14,14))
sns.set_style("whitegrid")
sns.barplot(x=df_fund_country_sorted.index, y=df_fund_country_sorted["lender_count"], color="grey",data=df_fund_country_sorted, ax=ax1)

ax1.xaxis.set_tick_params(rotation=90, labelsize=9)
ax1.set_ylabel("Median Anzahl Darlehensgeber, alle Projekte (graue Balken)", fontsize=14)
ax1.set_xlabel("Länder", fontsize=14)

ax2 = sns.lineplot(x=df_succ_sect_sorted.index, y=df_succ_sect_sorted["lender_count"], data=df_succ_sect_sorted, color="lightcoral")
ax3 = sns.lineplot(x=df_no_succ_sect_sorted.index, y=df_no_succ_sect_sorted["lender_count"], data=df_no_succ_sect_sorted, color="darkred")

ax2.set_ylabel('Median Anzahl Darlehensgeber | alle Projekte (Balken) vs. Projekte Erfolgsquote <= 50% (Linie)', fontsize=14)

plt.show()


# In[ ]:


# Elfenbeinküste, Mauretanien, Buthan und Afghanistan entfernen, da sie bis zu 17 mal so viele Darlehensgeber hat, 
# wie die anderen 95% der Länder und somit die Graphik stark verzerrt. Die Entferrnung hat keinen Einfluss auf 
# die Aussage der Graphik

df_final = df_final.loc[df_final["country_code"]!="CI",:]
df_final = df_final.loc[df_final["country_code"]!="MR",:]
df_final = df_final.loc[df_final["country_code"]!="BT",:]
df_final = df_final.loc[df_final["country_code"]!="AF",:]


# In[ ]:


# DataFrame df_success
# Elfenbeinküste, Mauretanien, Buthan und Afghanistan entfernen, da sie bis zu 17 mal so viele Darlehensgeber hat, 
# wie die anderen 95% der Länder und somit die Graphik stark verzerrt. Die Entferrnung hat keinen Einfluss auf 
# die Aussage der Graphik

df_success = df_success.loc[df_success["country_code"]!="CI",:]
df_success = df_success.loc[df_success["country_code"]!="MR",:]
df_success = df_success.loc[df_success["country_code"]!="BT",:]
df_success = df_success.loc[df_success["country_code"]!="AF",:]


# In[ ]:


# DataFrame df_success_medium
# Elfenbeinküste, Mauretanien, Buthan und Afghanistan entfernen, da sie bis zu 17 mal so viele Darlehensgeber hat, 
# wie die anderen 95% der Länder und somit die Graphik stark verzerrt. Die Entferrnung hat keinen Einfluss auf 
# die Aussage der Graphik

df_success_medium = df_success_medium.loc[df_success_medium["country_code"]!="CI",:]
df_success_medium = df_success_medium.loc[df_success_medium["country_code"]!="MR",:]
df_success_medium = df_success_medium.loc[df_success_medium["country_code"]!="BT",:]
df_success_medium = df_success_medium.loc[df_success_medium["country_code"]!="AF",:]


# In[ ]:


# DataFrame df_no_success
# Elfenbeinküste, Mauretanien, Buthan und Afghanistan entfernen, da sie bis zu 17 mal so viele Darlehensgeber hat, 
# wie die anderen 95% der Länder und somit die Graphik stark verzerrt. Die Entferrnung hat keinen Einfluss auf 
# die Aussage der Graphik

df_no_success = df_no_success.loc[df_no_success["country_code"]!="CI",:]
df_no_success = df_no_success.loc[df_no_success["country_code"]!="MR",:]
df_no_success = df_no_success.loc[df_no_success["country_code"]!="BT",:]
df_no_success = df_no_success.loc[df_no_success["country_code"]!="AF",:]


# In[ ]:


# Daten für ax1 vorbereiten
df_fund_country = df_final.groupby("country").agg({"lender_count":np.mean})
df_fund_country_sorted = df_fund_country.sort_values(by=["lender_count"], ascending=False)
df_fund_country_sorted


# In[ ]:


# Daten für ax2_success vorbereiten

df_succ_sect = df_success.groupby("country").agg({"lender_count":np.mean})
df_succ_sect_sorted = df_succ_sect.sort_values(by=["lender_count"], ascending=False)
df_succ_sect_sorted


# In[ ]:


# Daten für ax2 vorbereiten

df_no_succ_sect = df_no_success.groupby("country").agg({"lender_count":np.median})
df_no_succ_sect_sorted = df_no_succ_sect.sort_values(by=["lender_count"], ascending=False)
df_no_succ_sect_sorted


# In[ ]:


fig, ax1 = plt.subplots(figsize=(14,14))
sns.set_style("whitegrid")
sns.barplot(x=df_fund_country_sorted.index, y=df_fund_country_sorted["lender_count"], color="wheat",data=df_fund_country_sorted, ax=ax1)

ax1.xaxis.set_tick_params(rotation=90, labelsize=9)
ax1.set_ylabel("Median Anzahl Darlehensgeber, alle Projekte (graue Balken)", fontsize=14)
ax1.set_xlabel("Länder", fontsize=14)

#ax2 = ax1.twinx() # Verzerrt aufgrund der unterschiedlichen Skalierung die Darstellung.
ax2 = sns.lineplot(x=df_succ_sect_sorted.index, y=df_succ_sect_sorted["lender_count"], data=df_succ_sect_sorted, color="goldenrod")
ax2.set_ylabel('Median Anzahl Darlehensgeber | alle Projekte (Balken) vs. Projekte Erfolgsquote <= 50% (Linie)', fontsize=14)

#ax2 = ax1.twinx() # Verzerrt aufgrund der unterschiedlichen Skalierung die Darstellung.
ax3 = sns.lineplot(x=df_no_succ_sect_sorted.index, y=df_no_succ_sect_sorted["lender_count"], data=df_no_succ_sect_sorted, color="goldenrod")
ax3.set_ylabel('Median Anzahl Darlehensgeber | alle Projekte (Balken) vs. Projekte Erfolgsquote <= 50% (Linie)', fontsize=14)


plt.show()


# In[ ]:


# Daten für ax2 vorbereiten

df_no_succ_sect = df_no_success.groupby("country").agg({"lender_count":np.mean})
df_no_succ_sect_sorted = df_no_succ_sect.sort_values(by=["lender_count"], ascending=True)
df_no_succ_sect_sorted


# In[ ]:


# Daten für ax1 vorbereiten
df_fund_country = df_final.groupby("country").agg({"lender_count":np.mean})
# df_fund_country_sorted = df_fund_country.sort_values(by=["lender_count"], ascending=True)
# df_fund_country_sorted

# Daten für ax2 vorbereiten
df_no_succ_sect = df_no_success.groupby("country").agg({"lender_count":np.mean})
#df_no_succ_sect_sorted = df_no_succ_sect.sort_values(by=["lender_count"], ascending=False)
#df_no_succ_sect_sorted

# Plotten
fig, ax1 = plt.subplots(figsize=(14,14))
sns.set_style("whitegrid")
sns.barplot(x=df_fund_country_sorted.index, y=df_fund_country_sorted["lender_count"], color="wheat",data=df_fund_country_sorted, ax=ax1)

ax1.xaxis.set_tick_params(rotation=90, labelsize=9)
ax1.set_ylabel("Median Anzahl Darlehensgeber, alle Projekte (graue Balken)", fontsize=14)
ax1.set_xlabel("Länder", fontsize=14)

#ax2 = ax1.twinx() # Verzerrt aufgrund der unterschiedlichen Skalierung die Darstellung.
ax2 = sns.lineplot(x=df_no_succ_sect.index, y=df_no_succ_sect["lender_count"], data=df_no_succ_sect, color="goldenrod")
ax2.set_ylabel('Median Anzahl Darlehensgeber | alle Projekte (Balken) vs. Projekte Erfolgsquote <= 50% (Linie)', fontsize=14)

#ax3 = sns.lineplot(x=df_succ_sect_sorted.index, y=df_succ_sect_sorted["lender_count"], data=df_succ_sect_sorted, color="darkred")

plt.show()


# ##### Fragestellung bei Erstellung der Graphik: 
# 
# Überzeugen weniger erfolgreiche Teams auch weniger die Investoren? 
# Gibt je Land einen Unterschied bzgl. Anzahl der Darlehensgeber bei weniger "erfolgreichen" Teams (bzgl. Darlehenserhalt) gegenüber allen Teams? 
# 
# 
# ##### Erkenntnisse
# 
# - Weniger erfolgreiche Teams haben in fast allen Ländern __deutlich weniger Darlehensgeber__. Dies scheint schlüssig, da sie ggf. weniger Personen für Ihre Idee gewinnen konnten. Ein anderer Grund könnte sein, dass mehr Personen eher in der Lage sind, eine Idee angemessen zu bewerten. 
# 
# - Die __Unterschiede je Land sind sehr hetegerogen__. Hier könnte weitere Analyse, z.B. spielt höchstwahrscheinlich die Darlehenshöhe hier wiederum eine Rollen. Eine Betrachtung je Kontinent und somit kulturelle Hintergründe könnte weiteren auf Schluss geben.

# ## Inwieweit beinflusst Team & Zahlungsmodalitäten die Darlehenshöhe?

# In[ ]:


# Datensatzkopie erstellen

df_facetplot = df_final.copy()


# In[ ]:


# Test, ob erfolgreich

df_facetplot.info()


# In[ ]:


# nicht notwendige Zeilen entfernen

df_facetplot_drop =df_facetplot.drop(["loan_amount", "activity", "sector", "country", "currency", "country_code", "term_in_months", "lender_count", "female", "male"], axis=1)


# In[ ]:


# Test, ob erfolgreich

df_facetplot_drop.info()


# In[ ]:


# Zeilen mit NaN's löschen
# NaN's existieren nur in den Spalten team_count, female, male, sex_majority und team_category und stellen 
# mit 4221 NaNs nur 0,62% der Gesamtdaten dar. 
# Da für die folgende Graphike besonders "Team-Merkmale" (team_category und sex_majority) wichtig sind, 
# wurde beschlossen für den DataFrame für diese Graphik die NaNs zu löschen.

df_facetplot_oNaN = df_facetplot_drop.dropna(0)


# In[ ]:


# "Downcasting" von float Datensätzen

df_facetplot_oNaN["success_ratio"] = df_facetplot_oNaN["success_ratio"].apply(pd.to_numeric, downcast="float")


# In[ ]:


# Kontrolle, ob erfolgreich

df_facetplot_oNaN.info()


# In[ ]:


# Daten groupieren, damit sie im Plot leicht geladen werden können.

df_team_credit_cond = df_facetplot_oNaN.groupby(["sex_majority", "team_category", "repayment_interval"]).agg({"funded_amount":np.size})


# In[ ]:


# Test, ob erfolgreich

df_team_credit_cond


# In[ ]:


# Spaltennamen "auf eine Ebene" bringen

df_team_credit_cond.reset_index(inplace=True)


# In[ ]:


# Test, ob erfolgreich

df_team_credit_cond.head(2)


# In[ ]:


fig = px.bar(df_team_credit_cond, x="team_category", y="funded_amount", color="sex_majority", barmode="group", facet_col="repayment_interval", 
       category_orders={"repayment_interval": ["bullet", "irregular", "monthly", "weekly"]})
fig.show()


# ##### Fragestellung bei Erstellung der Graphik: 
# 
# Welche Zusammenhänge gibt es zwischen Teamgröße, Geschlechterverteilung, Auszahlungmodaliäten und der Darlehenshöhe?
# 
# 
# ##### Erkenntnisse
# 
# - Frauen scheinen mehr individuelle Bedingungen zu erhalten, während Männer eher einen monatlichen Kreditrückzahlungsmodus oder einen Bullet-Kredit erhalten. 
# - Bulletkredite gehen weniger an Einzelpersonen als bei den anderen Kreditformen. Bulletkredite werden eher für Gründungen mit unregelmäßigen Rückflüssen verwendet. Dies könnte für komplexere Projekte sprechen, bei denen  spezifischen Fachwissen und somit mehr Personen von Nöten sind.  

# # Anhang - weitere Visualisierungen

# ## Verschiedene Plots bzgl. Erfolgsquote

# In[ ]:


# Scatterplot für Zusammenhang Erfolgsquote/Funding Höhe mit lendercount, teams größe und Zusammensetzung


# In[ ]:


df_final.info()


# In[ ]:


df_final.describe()


# Datenvorbereitung:
# 
# Aus dem Data Preprocessing Pairplot und der deskriptiven Statistik sehen wir, dass ...

# In[ ]:


# DataFrame vorbereiten: funded_amount reduzieren

df_success_team = df_final.loc[(df_final["funded_amount"]<=10000) & (df_final["loan_amount"]<=10000),:]


# In[ ]:


# Test of erfolgreich

df_success_team.describe()


# In[ ]:


# Scatterplot erstellen

size = df_success_team["team_count"]**5

f, ax = plt.subplots(figsize=(8, 8))
sns.set_theme(style="whitegrid")

sns.scatterplot(x="loan_amount", y="funded_amount",
                hue="sex_majority", size=size,
                palette="deep",
                linewidth=0,
                data=df_success_team, ax=ax)

plt.show()


# ##### Erkenntnisse
# 
# Duch die hohe Menge der Projekte, macht Darstellung wenig Sinn.
# 
# Mögliche Lösugn: Entertainment untersuchen, da niedrigste Erfolgsquote und zweitkleinste Segment

# In[ ]:


# DataFrame vorbereiten

df_success_team_ent = df_success_team.loc[(df_success_team["sector"]=="Entertainment")]
df_success_team_ent


# In[ ]:


# Scatterplot erstellen

size = df_success_team_ent["team_count"]**5

f, ax = plt.subplots(figsize=(8, 8))
sns.set_theme(style="whitegrid")

sns.scatterplot(x="loan_amount", y="funded_amount",
                hue="team_category", size=size,
                palette="deep",
                linewidth=0,
                data=df_success_team_ent, ax=ax)

plt.show()


# ##### Erkenntnisse
# 
# Die Team Kategorie bringt mehr Informationen. Alle Projekte, die nicht oder nur teilweis Darlehnen erhalten haben, waren ausschliesslich one (wo)man shows.
# 
# Die Größe bietet bisher aufgrund der Menge der Fälle und der ungleichen Verteilung bisher keinen Mehrwert
# 
# Nun Versuch mit etwas mehr Fällen. Manufacturing ist etwas größer, jedoch noch nicht so groß

# In[ ]:


# DataFrame vorbereiten

df_success_team_man = df_success_team.loc[(df_success_team["sector"]=="Housing")]
df_success_team_man


# In[ ]:


# Scatterplot erstellen

size = df_success_team_man["team_count"]**5

f, ax = plt.subplots(figsize=(8, 8))
sns.set_theme(style="whitegrid")

sns.scatterplot(x="loan_amount", y="funded_amount",
                hue="team_category", size=size,
                palette="deep",
                linewidth=0,
                data=df_success_team_man, ax=ax)

plt.show()


# ## Funding

# ### Anzahl Fundingprojekte bzgl. Dauer Kreditauszahlung

# In[ ]:


# DataFrame für Verteilung erstellen
df_term_in_months = df_final.groupby(["term_in_months"]).size()

# Barplot erstellen
fig = px.bar(df_term_in_months, 
             labels={'value':'Anzahl Fundingprojekte', 'term_in_months':'Dauer Kreditauszahlung in Monaten'}, 
             height=500)
fig.show()


# ##### Erkenntnisse: 
# - Der Großteil der Funding Projekte wird nicht über einen längeren Zeitraum als 15 Monate ausgezahlt.
# - 14 Monate und 8 Monate sind mit großem Abstand die am häufigsten gewählten Zeiträume.
# - Mehr als 27 Monate wird nur bei wenigen Ausnahmen ausgezahlt. den wenigsten  ausgezahlt

# ### Kumulierte Fundinghöhe je Country

# In[ ]:


# DataFrame für Verteilung erstellen

df_funding_country = df_final.groupby("country_code").agg({"funded_amount":np.sum})
df_fund_contry_sorted = df_funding_country.sort_values(by=["funded_amount"], ascending=False)


# In[ ]:


# Barplot erstellen

fig = px.bar(df_fund_contry_sorted)
fig.show()


# ### Durchschnittliche Fundinghöhe je Country

# In[ ]:


# DataFrame für Verteilung erstellen

df_funding_country = df_final.groupby("country_code").agg({"funded_amount":np.mean})
df_fund_contry_sorted = df_funding_country.sort_values(by=["funded_amount"], ascending=False)


# In[ ]:


# Barplot erstellen

fig = px.bar(df_fund_contry_sorted)
fig.show()


# ### Kumulierte Funding Höhe je Sector

# In[ ]:


# DataFrame für Verteilung erstellen

df_funding_sector = df_final.groupby("sector").agg({"funded_amount":np.sum})
df_fund_sector_sorted = df_funding_sector.sort_values(by=["funded_amount"], ascending=False)
df_fund_sector_sorted


# In[ ]:


# Barplot erstellen

fig = px.bar(df_fund_sector_sorted)
fig.show()


# ## Erfolgsquote

# ### Erfolgsquote je Sektor

# In[ ]:


# DataFrame für Verteilung erstellen

df_success_ratio_sector = df_final.groupby("sector").agg({"success_ratio":np.mean})
df_success_ratio_sector


# In[ ]:


# Barplot erstellen
fig = px.bar(df_success_ratio_sector, 
             labels={'value':'Mittelwert:<br>Verhältnis loan_amount zu funded_amount', 'country_code':'Länder'}, 
             height=500).update_xaxes(categoryorder="total ascending")
fig.show()


# ### Erfolgsquote je Land

# In[ ]:


# DataFrame für Verteilung erstellen

df_success_ratio = df_final.groupby("country_code").agg({"success_ratio":np.mean})
df_success_ratio


# In[ ]:


# Barplot erstellen
fig = px.bar(df_success_ratio, 
             labels={'value':'Mittelwert:<br>Verhältnis loan_amount zu funded_amount', 'country_code':'Länder'}, 
             height=500).update_xaxes(categoryorder="total ascending")
fig.show()


# ##### Erkenntnisse:
# Der Großteil der Projekte hat über 90% der gewünschten Fundingsumme erhalten

# ### Nicht erfolgreiche Projekte

# In[ ]:


# DataFrame erstellen mit wenig erfolgreichen erstellen (Erfolgsquote kleiner gleich fünfzig)

df_no_success = df_final.loc[(df_final["success_ratio"]<=50),:]
df_no_success


# #### Bzgl. Teamgröße

# In[ ]:


df_no_success_group = df_no_success.groupby("team_category").size()

fig = px.bar(df_no_success_group).update_xaxes(categoryorder="total ascending")
fig.show()


# #### Bzgl. Geschlecht

# In[ ]:


df_no_success_sex = df_no_success.groupby("sex_majority").size()

fig = px.bar(df_no_success_sex,
            labels={'value':'numer of funding projects', 'sex_majority':'sex majority per team'}, 
             height=500
            ).update_xaxes(categoryorder="total ascending")
fig.show()


# ## Sektoren

# ### Erfolgsquote je Sektor und Teamgröße

# In[ ]:


# DataFrame erstellen

df_sector_team_cat = df_final.groupby(["sector", "team_category"], as_index=False).agg({"success_ratio":np.mean})
df_sector_team_cat


# In[ ]:


df_sector_team_cat.columns


# In[ ]:


fig = px.bar(df_sector_team_cat, x="sector", y="success_ratio", color="team_category", barmode="group")
fig.show()


# ### Sektor bzgl. Darlehenshöhe & Geber - Mittelwert

# In[ ]:


# Daten für ax1 vorbereiten

df_funding_sector_avg = df_final.groupby("sector").agg({"funded_amount":np.mean})
df_fund_sector_avg_sorted = df_funding_sector_avg.sort_values(by=["funded_amount"], ascending=False)
df_fund_sector_avg_sorted


# In[ ]:


# Daten für ax2 vorbereiten

df_funding_lender_avg = df_final.groupby("sector").agg({"lender_count":np.mean})
df_funding_lender_avg_sorted = df_funding_lender_avg.sort_values(by=["lender_count"], ascending=False)
df_funding_lender_avg_sorted


# In[ ]:


# Plot erstellen
 
fig, ax1 = plt.subplots(figsize=(10,10))
sns.set_style("whitegrid")
sns.barplot(x=df_fund_sector_avg_sorted.index, y=df_fund_sector_avg_sorted["funded_amount"], palette="Blues_r",data=df_fund_sector_avg_sorted, ax=ax1)

ax1.xaxis.set_tick_params(rotation=70, labelsize=10)
ax1.set_ylabel("Durchschnittliche Höhe Darlehen", fontsize=14)
ax1.set_xlabel("Sektor", fontsize=14)

ax2 = ax1.twinx()
ax2 = sns.lineplot(x=df_funding_lender_avg_sorted.index, y=df_funding_lender_avg_sorted["lender_count"], data=df_funding_lender_avg_sorted, color="darkred")
ax2.set_ylabel('Durchschnitt Anzahl Darlehensgeber', fontsize=14)

plt.show()


# ## Team Verteilungen

# ### Geschlechterverhältnis 

# In[ ]:


df_all_sex = df_final.groupby("sex_majority").size()

fig = px.bar(df_all_sex,
            labels={'value':'numer of funding projects', 'sex_majority':'sex majority per team'}, 
             height=500
            ).update_xaxes(categoryorder="total ascending")
fig.show()


# ### team_member_count

# In[ ]:


# DataFrame für Verteilung erstellen
df_team_member = df_final.groupby("team_count").size()

# Barplot erstellen
fig = px.bar(df_team_member,
             labels={'value':'Anzahl Funding Projekte', 'team_member_count':'Anzahl Teammitglieder'}, 
             height=500).update_xaxes(categoryorder="total ascending")
fig.show()


# ##### Erkenntnisse:
# Der größte Teil der Funding Projekte, werden von einer Person durchgeführt.
# 

# ### Teamgröße und Erfolgsquote

# In[ ]:


df_sector_team_cat = df_final.groupby(["sector", "team_category"], as_index=False).agg({"success_ratio":np.mean})
df_sector_team_cat


# In[ ]:


df_success_team = df_final.groupby("team_category").agg({"success_ratio":np.mean})
df_success_team


# In[ ]:


fig = px.bar(df_success_team)
fig.show()


# ## Kreditkonditionen

# ### repayment_interval

# In[ ]:


# Ausprägungen und Häufigkeiten zu repayment_interval erhalten

df_final.groupby("repayment_interval").size()


# In[ ]:


# Zusammenhang bzgl. funded_amount und repayment_interval betrachten
# DataFrame anlegen

df_repayment_sorted = df_final.groupby("repayment_interval").agg({"funded_amount":np.mean}).sort_values(by=["funded_amount"], ascending=False)
df_repayment_sorted


# In[ ]:


df_repayment_sorted.index


# In[ ]:


# Barplot erstellen

fig = px.bar(df_repayment_sorted)
fig.show()


# In[ ]:


# Pieplot erstellen

fig = px.pie(df_repayment_sorted, values='funded_amount', names=['monthly', 'bullet', 'irregular', 'weekly'], title='payment interval')
fig.show()


# ##### Erkenntnis: 
# Weekly interval kommen wie erwartet nur für kleinere Kredite in Betracht

# In[ ]:


# Daten für ax1 vorbereiten

df_funding_country = df_final.groupby("country").agg({"funded_amount":np.median})
df_fund_country_sorted = df_funding_country.sort_values(by=["funded_amount"], ascending=True)
df_fund_country_sorted

# Daten für ax2 vorbereiten

df_funding_lender = df_final.groupby("country").agg({"lender_count":np.median})
df_funding_lender_sorted = df_funding_lender.sort_values(by=["lender_count"], ascending=True)
df_funding_lender_sorted

# Plot erstellen
 
fig, ax1 = plt.subplots(figsize=(10,10))
sns.set_style("whitegrid")
sns.barplot(x=df_fund_country_sorted.index, y=df_fund_country_sorted["funded_amount"], palette="Blues",data=df_fund_country_sorted, ax=ax1)

ax1.xaxis.set_tick_params(rotation=70, labelsize=10)
ax1.set_ylabel("Median Höhe Darlehen", fontsize=14)
ax1.set_xlabel("Länder", fontsize=14)

ax2 = ax1.twinx()
ax2 = sns.lineplot(x=df_funding_lender_sorted.index, y=df_funding_lender_sorted["lender_count"], data=df_funding_lender_sorted, color="darkred")
ax2.set_ylabel('Median Anzahl Darlehensgeber', fontsize=14)

plt.show()


# In[ ]:


# ax1
df_sector = df_final.groupby("sector").agg({"funded_amount":np.sum})
df_sector_sorted_funded = df_sector.sort_values(by=["funded_amount"], ascending=True)

# ax2
df_sector = df_final.groupby("sector").agg({"sector":np.size})
df_sector = df_sector.rename(columns={"sector": "amount_projects"})
df_sector = df_sector_rename.reset_index()
df_sector_sorted_amount = df_sector.sort_values(by=["amount_projects"])

# Plot erstellen
 
fig, ax1 = plt.subplots(figsize=(10,10))
sns.set_style("whitegrid")
sns.barplot(x=df_sector_sorted_funded.index, y=df_sector_sorted_funded["funded_amount"], palette="Blues",data=df_sector_sorted_funded, ax=ax1)

ax1.xaxis.set_tick_params(rotation=70, labelsize=10)
ax1.set_ylabel("Durchschnitt Darlehenshöhe in US-Dollar (Balken)", fontsize=14)
ax1.set_xlabel("Sektor", fontsize=14)

ax2 = ax1.twinx()
ax2 = sns.lineplot(x=df_sector_sorted_amount.index, y=df_sector_sorted_amount["amount_projects"], data=df_sector_sorted_amount, color="darkred")
ax2.set_ylabel('Durchschnitt Anzahl Darlehensgeber (Linie)', fontsize=14)

plt.show()


# In[ ]:


# nach Funding-Summe

df_cont = df_final.groupby("continent").agg({"continent":np.size, "funded_amount":np.sum})
df_cont_rename = df_cont.rename(columns={"continent": "amount_projects"})
df_cont = df_cont_rename.reset_index()
df_cont_sorted = df_cont.sort_values(by=["amount_projects"])
#df_cont_sorted_fund = df_cont.sort_values(by=["funded_amount"])

sns.set(rc={'figure.figsize':(10,8)})
sns.set_theme(style="whitegrid")
ax = sns.barplot(x="continent", y="funded_amount", data=df_cont_sorted)


# In[ ]:


fig = px.bar(df_cont_count_sorted, x="amount_projects", y="country", barmode="group")
fig.update_layout(
    font_size = 8,
    autosize=False,
    width=900,
    height=1000,
    margin=dict(
        l=50,
        r=50,
        b=100,
        t=100,
        pad=4
    ),
)
fig.show()


# In[ ]:


# Daten für ax1 vorbereiten

df_sector = df_final.groupby("sector").agg({"sector":np.size, "funded_amount":np.sum})
df_sector = df_sector.rename(columns={"sector": "amount_projects"})
df_sector_sum = df_sector.reset_index()

# Daten für ax2 vorbereiten

df_sector = df_final.groupby("sector").agg({"sector":np.size, "funded_amount":np.mean})
df_sector = df_sector.rename(columns={"sector": "amount_projects"})
df_sector_mean = df_sector.reset_index()

# Plot erstellen
 
fig, ax1 = plt.subplots(figsize=(10,10))
sns.set_style("whitegrid")
sns.barplot(x=df_sector_sum.index, y=df_sector_sum["funded_amount"], palette="Blues",data=df_sector_sum, ax=ax1)

ax1.xaxis.set_tick_params(rotation=70, labelsize=10)
ax1.set_ylabel("Absolute Fundinghöhe in US-Dollar (Balken)", fontsize=14)
ax1.set_xlabel("Sektor", fontsize=14)

ax2 = ax1.twinx()
ax2 = sns.lineplot(x=df_sector_mean["sector"], y=df_sector_mean["funded_amount"], data=df_sector_mean, color="darkred")
ax2.set_ylabel('Durchschnitt Fundinghöhe in US-Dollar (Linie)', fontsize=14)

plt.show()


# In[ ]:


df_cont_count = df_dropna_funding.groupby(["continent", "country"]).agg({"continent":np.size, "funded_amount":np.sum})
df_cont_count_rename = df_cont_count.rename(columns={"continent": "amount_projects"})
df_cont_count = df_cont_count_rename.reset_index()
df_cont_count = df_cont_count.sort_values(by=["amount_projects"])
df_cont_count = df_cont_count.loc[df_cont_count["funded_amount"]!=0, :]
df_cont_count

fig = px.scatter(df_cont_count, x="amount_projects", y="funded_amount", color="continent",
           hover_name="country", log_x=True, size_max=60)
fig.show()


# # Learnings

# Bei Balkendiagrammen auf gleiche Reihenfolge achten
