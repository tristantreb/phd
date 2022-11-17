import pandas as pd
import plotly.express as px


# Create a scatter plot with px with subsampled x data and y
def plot_subsampled_scatter(x, y, O2_FEV1, ex_column='Is Exacerbated', ex_color='rgba(213,094,000,0.7)', stable_color='rgba(000,114,178,0.7)', random_state=42):
  # Subsample
  ex = O2_FEV1[O2_FEV1[ex_column] == True]
  non_ex = O2_FEV1[O2_FEV1[ex_column] == False]
  subsample_factor = round(ex.shape[0])
  O2_FEV1_sub=pd.concat([ex, non_ex.sample(subsample_factor, random_state=random_state)], axis=0)

  fig = px.scatter(O2_FEV1_sub, x=x, y=y, color=ex_column, color_discrete_map={True: ex_color, False: stable_color})
  fig.update_layout(height=400, width=1150, title='Subsampled measurement done in stable period'.format(x, y))
  return fig