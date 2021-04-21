from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score

def display_images(csv, rows, cols, show=True, title_column=None, fname=None):
    """Plots grid of random images

    Parameters
    ----------
    csv : pd.DataFrame
        DataFrame with images relative pathes in 'path' column
    rows : int
        Number of rows in the grid 
    cols : int
        Number of columns in the grid
    show : bool, optional
        Show the plot, by default True
    title_column : str, optional
        The name of column with title description, by default None
        If None, doesn't show the title. 
    fname : str, optional
        The name of file with saved plot, by default None
        If not None, saves plot, else doesn't.
    """    
    
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(8, 8), dpi=150)
    n_total = len(csv)
    
    n_grid = rows * cols
    subset = csv.sample(n=n_grid, replace=n_grid > n_total)
    
    axes = axes.ravel() if n_grid > 1 else [axes]
    
    i = 0
    
    for index, row in subset.iterrows():
        image = Image.open(row.path)
        
        axes[i].imshow(image)
        if title_column:
            title = row[title_column]
            #title = "\n".join(title.split())
            axes[i].set_title(title, fontsize=10)
        axes[i].set_axis_off()
        axes[i].imshow(image)
        axes[i].set_axis_off()
        
        i += 1
        
    if fname is not None:
        plt.savefig(fname, dpi=150)
    if show:
        #plt.tight_layout()
        plt.show()
    plt.close(fig)

def display_metrics(labels, preds, threshold):

    precision = precision_score(labels, preds > threshold)
    recall = recall_score(labels, preds > threshold)
    f1 = f1_score(labels, preds > threshold)
    conf_matrix = confusion_matrix(labels, preds > threshold)
    print(f"Precision = {precision:.4f}")
    print(f"Recall = {recall:.4f}")
    print(f"F1 = {f1:.4f}")
    print("Confusion matrix:")
    print(conf_matrix)