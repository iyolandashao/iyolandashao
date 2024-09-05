import redshift_connector
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

REDSHIFT_HOST = ''
REDSHIFT_USER = ''
REDSHIFT_DB = ''
REDSHIFT_PASSWORD = ''
REDSHIFT_PORT = ''

conn = redshift_connector.connect(host=REDSHIFT_HOST, database=REDSHIFT_DB, user=REDSHIFT_USER, password=REDSHIFT_PASSWORD)
df_viz = pd.read_sql_query('select * from sandbox.viz', conn)
df_viz.columns = ['contact_id', 'lead_timestamp', 'sal_timestamp', 'sql_timestamp', 'trial_timestamp', 'customer_timestamp', 'deal_reach_out_via', 'country', 'region', 'city', 'leads', 'sal', 'sql', 'trials', 'newcustomers']
df0_viz = df_viz.copy()
df0_viz = df0_viz[df0_viz['newcustomers']==1]

#all
df00_viz = pd.read_csv('df_new.csv')
df00_viz = df00_viz[['customers', 'conversion_time', 'employees_off', 'revenue']]
df_test = df00_viz
df_test = StandardScaler().fit_transform(df_test)
df_test.shape
np.mean(df_test),np.std(df_test)
feat_cols = ['feature'+str(i) for i in range(df_test.shape[1])]
normalised_off = pd.DataFrame(df_test,columns=feat_cols)
normalised_off.columns = ['newcustomers', 'conversion_time', 'employees_off', 'revenue']
normalised_off.corr()
sns.heatmap(normalised_off.corr())
plt.figure(figsize=(16, 6))
# define the mask to set the values in the upper triangle to True
mask = np.triu(np.ones_like(normalised_off.corr(), dtype=np.bool))
heatmap = sns.heatmap(normalised_off.corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Correlation Heatmap of revenue factors', fontdict={'fontsize':18}, pad=16);
plt.tight_layout()
plt.show()
plt.savefig('Correlation Heatmap_revenue'+'.png')
#sales
df00_viz = pd.read_csv('df_new.csv')
df00_viz = df00_viz[['customers', 'conversion_time', 'sales_off', 'revenue']]
df_test = df00_viz
df_test = StandardScaler().fit_transform(df_test)
df_test.shape
np.mean(df_test),np.std(df_test)
feat_cols = ['feature'+str(i) for i in range(df_test.shape[1])]
normalised_off = pd.DataFrame(df_test,columns=feat_cols)
normalised_off.columns = ['newcustomers', 'conversion_time', 'sales_off', 'revenue']
normalised_off.corr()
sns.heatmap(normalised_off.corr())
plt.figure(figsize=(16, 6))
# define the mask to set the values in the upper triangle to True
mask = np.triu(np.ones_like(normalised_off.corr(), dtype=np.bool))
heatmap = sns.heatmap(normalised_off.corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Correlation Heatmap of revenue_sales', fontdict={'fontsize':18}, pad=16);
plt.tight_layout()
plt.show()
plt.savefig('Correlation Heatmap_revenue_sales'+'.png')
