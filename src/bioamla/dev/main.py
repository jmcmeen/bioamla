import pandas as pd
from renumics import spotlight

df = pd.read_csv("scp2023_ast_esc50.csv")

# df = df.groupby('prediction').agg({'prediction': 'count'})

prediction_counts = df['prediction'].value_counts().to_frame('count').reset_index()
prediction_counts.columns = ['prediction', 'count']

prediction_counts.to_csv("scp2023_ast_esc50_prediction_counts.csv", index=False)


# spotlight.show(prediction_counts)