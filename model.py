import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
path = r'C:\Users\Sark\VSCODE\Beginner\.venv\PROJECT\Dataset\*csv' 
all_files = glob.glob(path)
df_list = []
for filename in all_files:
 
    df = pd.read_csv(filename, index_col=None, header=0)

    df_list.append(df)
master_df = pd.concat(df_list, axis=0, ignore_index=True)
target_col = 'FORCE_2020_LITHOFACIES_LITHOLOGY'


master_df.dropna(subset=[target_col], inplace=True)
master_df.reset_index(drop=True,inplace=True)
# print(master_df.head())

X=master_df.drop(columns=['FORCE_2020_LITHOFACIES_LITHOLOGY'])
y=master_df['FORCE_2020_LITHOFACIES_LITHOLOGY']
# print(X.shape)
# print(y.shape)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
numerical_cols = X_train.select_dtypes(include=['number']).columns
X_train = X_train[numerical_cols]
X_test = X_test[numerical_cols]
X_train = X_train.dropna(axis=1, how='all')
X_test = X_test.dropna(axis=1, how='all')
train_medians=X_train.median()
X_train = X_train.fillna(train_medians)
X_test = X_test.fillna(train_medians)
model=RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=42)
model.fit(X_train,y_train)

y_pred=model.predict(X_test)
# print(y_pred)
results = master_df.loc[X_test.index, ['WELL', 'DEPTH_MD']].copy()
results['Real_Lithology'] = y_test
results['Model_Prediction'] = y_pred
print(results.head(10))
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# 1. PREPARE THE DATA
# We need to grab the Depth and GR from the original master_df using the index from X_test
# (because X_test might have lost the Depth column during training)
# plot_indices = X_test.index[:500] # Let's just plot the first 500 rows to keep it zoomed in
# df_plot = master_df.loc[plot_indices].copy()

# # Add your model's predictions to this slice
# df_plot['Prediction'] = model.predict(X_test.loc[plot_indices])
# depth_col='DEPTH_MD'
# df_plot=df_plot.sort_values(by=depth_col)

# # 2. SETUP THE PLOT
# depth_col = 'DEPTH_MD' # Check your column name!
# gr_col = 'GR'
# lith_col = 'Prediction'

# # Define Colors (Sand=Yellow, Shale=Green)
# # Adjust these numbers to match your specific Lithology Codes!
# lith_color_map = {
#     30000: '#FFFF00', # Sandstone
#     65000: '#228B22', # Shale
#     99000: '#BEBEBE', # Other
#     # Add more codes if your model predicts them
# }
# lith_codes = list(lith_color_map.keys())
# lith_colors = list(lith_color_map.values())
# cmap_facies = mcolors.ListedColormap(lith_colors)
# bounds = lith_codes + [max(lith_codes) + 1]
# norm = mcolors.BoundaryNorm(bounds, cmap_facies.N)
# # --- 3. DRAW IT ---
# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 12), sharey=True)
# fig.suptitle(f'My AI Model Results (Test Set)', fontsize=16)

# # Track 1: Gamma Ray (The Input)
# ax[0].plot(df_plot[gr_col], df_plot[depth_col], color='black', linewidth=0.5)
# ax[0].set_xlabel("Gamma Ray")
# ax[0].set_xlim(0, 150)
# ax[0].grid(which='major', color='lightgrey', linestyle='-')

# # THE FIX: Use tick_top() instead of set_position()
# ax[0].xaxis.tick_top()
# ax[0].xaxis.set_label_position('top')

# # Track 2: AI Prediction (The Output)
# lith_data = df_plot[lith_col].values
# lith_2d = np.expand_dims(lith_data, axis=1)
# ax[1].imshow(lith_2d, aspect='auto', cmap=cmap_facies, norm=norm,
#              extent=[0, 1, max(df_plot[depth_col]), min(df_plot[depth_col])])
# ax[1].set_xlabel("AI Class Prediction")
# ax[1].set_xticks([])

# # THE FIX: Use tick_top() here too
# ax[1].xaxis.tick_top()
# ax[1].xaxis.set_label_position('top')

# plt.ylim(max(df_plot[depth_col]), min(df_plot[depth_col]))
# plt.tight_layout()
plt.show()
 
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)
print(f"Final Model Accuracy: {score:.2%}")
print(y_test.value_counts())
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.title(f'Confusion Matrix (Accuracy: {score:.2%})')
plt.xlabel('Predicted Label (What the AI thought)')
plt.ylabel('True Label (What the Rock actually is)')
plt.show()
# import joblib
# joblib.dump(model, 'my_lithology_model.pkl')
# joblib.dump(train_medians, 'my_medians.pkl')


