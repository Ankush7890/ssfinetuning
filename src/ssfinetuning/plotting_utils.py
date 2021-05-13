import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import gc
from .training_args import generate_kwargs

def add_end_args(from_fn):
    
    addFromstr = f'Adding Args from function-> {from_fn.__name__}'
    def docstring_decorator(fn):
        index = from_fn.__doc__.find('Args:')
        fn.__doc__ = fn.__doc__ + addFromstr + from_fn.__doc__[index+5:]
        return fn

    return docstring_decorator

def sort_and_find(data, cols_unique_vals, x_axis_col, y_axis_col, select_best, criteria):
    
    """
    Finds a sorted list of "y_axis_col" in "data" based on the "criteria". First, this 
    function generates all the combinations of unique cols values and then it creates a 
    list of values for all the combinations. At the end, it sorts this list based on the criteria.  
    
    Args:
    
    data (:obj:`pd.DataFrame` ): The data to be sorted.
    
    cols_unique_vals (:obj:`list`): A list of columns of important hyperparameters 
    with their unique values.
    
    x_axis_col (:obj:`str`): Column name with the values to be plotted on the x axis.

    y_axis_col (:obj:`str`): Column name with the values to be plotted on the y axis.
    
    criteria (:obj:`str`): criteria to sort the list. There are three choices, 
    (i) max, (ii) min, and (iii)mean.
    
    Return:
    
    sorted_list (:obj:`str`): 
    """
 
    value_for_combi=[]
    for comb in generate_kwargs(cols_unique_vals):
        data_copy = data.copy()

        for k, v in comb.items():
            data_copy = data_copy[data_copy[k]==v]

        value_for_combi.append([comb,
                           data_copy[x_axis_col],
                           data_copy[y_axis_col], 
                           data_copy[y_axis_col].mean(),
                           data_copy[y_axis_col].max(), 
                           data_copy[y_axis_col].min()])
        del data_copy
        gc.collect()
        
    if select_best:
        
        if(criteria == 'mean'):
            sorted_list = sorted(value_for_combi, key=lambda x: x[3], reverse=True)

        elif(criteria == 'max'):
            sorted_list = sorted(value_for_combi, key=lambda x: x[4], reverse=True)

        elif(criteria == 'min'): 
            sorted_list = sorted(value_for_combi, key=lambda x: x[5], reverse=True)
    
    return sorted_list

def set_default_vals(num_graphs):

    """
    Sets the default values for font size, line widths, markers etc.
    
    Args:
    
    num_graphs (:obj:`int` ): num of num_graphs to be plotted with maximum value of 4.
    
    """
    
    if num_graphs==1:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(18,9))
        axes_unrolled = [axes]
        mpl.rcParams['lines.linewidth'] = 3.0
        mpl.rcParams['lines.markersize'] = 12.0
        mpl.rcParams['lines.markeredgewidth'] = 1.0
        mpl.rcParams['xtick.labelsize'] = 27
        mpl.rcParams['ytick.labelsize'] = 27
        mpl.rcParams['axes.labelsize'] = 34.5
        mpl.rcParams['font.size']=23
    
    elif num_graphs==2:
        
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 7))
        axes_unrolled = axes.reshape(1,2)[0]
        mpl.rcParams['lines.linewidth']=6.0
        mpl.rcParams['lines.markersize']=12.0
        mpl.rcParams['lines.markeredgewidth']=1.0
        mpl.rcParams['xtick.labelsize']=27
        mpl.rcParams['ytick.labelsize']=27
        mpl.rcParams['axes.labelsize']=34.5
        mpl.rcParams['font.size']=23
    
    else:
        
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(33, 22))
        axes_unrolled = axes.reshape(1,4)[0]
        mpl.rcParams['lines.linewidth'] = 6.6
        mpl.rcParams['lines.markersize'] = 22
        mpl.rcParams['lines.markeredgewidth'] = 2.2
        mpl.rcParams['xtick.labelsize'] = 44
        mpl.rcParams['ytick.labelsize'] = 44
        mpl.rcParams['axes.labelsize'] = 44
        mpl.rcParams['font.size'] = 33.2
    
    return fig, axes, axes_unrolled

def get_default_legend_pos(num_graphs, axes_index=None):
    
    """
    Sets the default values where legends could be placed.
    
    Args:
    
    num_graphs (:obj:`int` ): num of num_graphs to be plotted with maximum value of 4.
    
    axes_index (:obj:`int` , `optional`, defaults to None ): In the case of multiple, 
    setting changed depending on index of axes.
    
    """
    
    if num_graphs==1:
        kwargs={'bbox_to_anchor':(-0.4, 1.00), 'loc':2, 'borderaxespad':0.}
    elif num_graphs==2:
        x_displacement = -1.0 if axes_index%2==0 else 1.0
        kwargs={'bbox_to_anchor':(x_displacement, 1.00), 'loc':2, 'borderaxespad':0.}
    else:
        x_displacement = -0.7 if axes_index%2==0 else 1.0
        kwargs={'bbox_to_anchor':(x_displacement, 1.00), 'loc':2, 'borderaxespad':0.}
    
    return kwargs

def plot_in(axes, axes_index = 0, data=None,  data_to_compare=None, x_axis_col='epoch', 
            y_axis_col="eval_mc", select_best=5, criteria='max', cols_to_find=["w_ramprate"], 
            dis_col='l_fr', dis_val=False, data_to_compare_lb='sup_stats'):
    """
    Main plotting function.
    
    Args:
    
    axes (:obj:`matplotlib.pyplot.axes` ): axes object to plot.
    
    axes_index (:obj:`int`, `optional`, defaults to None ): In the case 
    of multiple, setting changed depending on index of axes.
    
    data (:obj:`pd.DataFrame` ): Data to sort from.
    
    data_to_compare (:obj:`pd.DataFrame` ): Data to compare with 
    the sorted results. For example, purely supervised results.
    
    x_axis_col (:obj:`str`, `optional`, defaults to 'epoch' ): Column name 
    with the values to be plotted on the x axis.
    
    y_axis_col (:obj:`str`, `optional`, defaults to 'eval_mc' ): Column name 
    with the values to be plotted on the y axis.
    
    select_best (:obj:`int`, `optional`, defaults to 5 ): The number of plots 
    to be made based out of the sorted list.
    
    criteria (:obj:`str`, `optional`, defaults to 'max' ): Criteria to sort the 
    list. There are three choices, (i) max, (ii) min, and (iii)mean.
    
    cols_to_find (:obj:`list`, `optional`, defaults to ['w_ramprate'] ): The list of 
    column names which will analysed to find the best of them based the sorting criteria.
    
    dis_col (:obj:`str`, `optional`, defaults to 'l_fr' ): The dicriminatory column name. 
    This would be column name along which the graphs would be divided along the subplots.
    
    dis_val (:obj:`int` or 'float', `optional`, defaults to 'None' ): This is only valid
    if the 'dis_col' is not None. This is used when a certain unique value of discriminatory 
    column is plotted.
    
    data_to_compare_lb (:obj:`str`, `optional`, defaults to 'sup_stats' ): Label name for 
    the data_to_compare plot. 
    
    """
 
    cols_unique_vals = { v : data[v].unique() for v in cols_to_find }
    sorted_list = sort_and_find(data, cols_unique_vals, x_axis_col, y_axis_col, select_best, criteria)

    plots_range = min(select_best, len(sorted_list))
    
    for i in range(plots_range):
        label_name = '\n'.join([str(k)+'='+str(v) for k, v in sorted_list[i][0].items()])
        axes.plot(sorted_list[i][1], sorted_list[i][2], label=label_name)

    if data_to_compare is not None:
        if dis_val:
            data_2c_dis = data_to_compare[data_to_compare[dis_col] == dis_val]
            axes.plot(data_2c_dis[x_axis_col], data_2c_dis[y_axis_col], label = data_to_compare_lb,linestyle='--')        
        else:
            axes.plot(data_to_compare[x_axis_col], data_to_compare[y_axis_col], label = data_to_compare_lb, linestyle='--')
    
    
    axes.set_xlabel(x_axis_col)
    if dis_val: axes.set_title(dis_col+'='+str(dis_val))
    if axes_index%2==0: axes.set_ylabel(y_axis_col)

@add_end_args(plot_in)
def plot_with_discriminator(dis_col, save_png, data=None, *args, **kwargs):

    """
    Plotter if discriminatory column is specified.
    
    Args:
    
    dis_col (:obj:`str` ): The dicriminatory column name. This would be column
    name along which the graphs would be divided along the subplots.
    
    save_png (:obj:`str` ): Whether to save png of results or not. If the 
    value of save_png is not None then it would save the image with name of 
    string value set in save_png.
    
    kwargs: remaining dictionary of keyword arguments from the plot_in function.

    """
    
    fig, axes, axes_unrolled = set_default_vals(len(data[dis_col].unique()))
        
    for dis_index, dis_val in enumerate(data[dis_col].unique()):
        
        data_dis = data[data[dis_col]==dis_val]
                
        plot_in(axes_unrolled[dis_index], axes_index=dis_index, data=data_dis, dis_val = dis_val, *args, **kwargs)
                
        axes_unrolled[dis_index].legend(**get_default_legend_pos(len(data[dis_col].unique()), dis_index))
        
    if save_png :plt.savefig(save_png, bbox_inches='tight')


@add_end_args(plot_in)
def simple_plot(save_png, *args, **kwargs):

    """
    Plotter if discriminatory column is not specified.
    
    Args:

    save_png (:obj:`str` ): Whether to save png of results or not. If the value of
    save_png is not None then it would save the image with name of string value set 
    in save_png.
    
    kwargs: remaining dictionary of keyword arguments from the plot_in function.

    """
        
    fig, axes, _ = set_default_vals(1)
        
    plot_in(axes, *args, **kwargs)
    
    axes.legend(**get_default_legend_pos(1))
    
    if save_png : plt.savefig(save_png, bbox_inches= 'tight')

@add_end_args(plot_in)
def sort_and_plot(dis_col=None, save_png='results.png', *args, **kwargs):

    """
    Function to sort the results and plot them depending if discriminatory is specified or not.
    
    Args:
    
    dis_col (:obj:`str`, `optional`, defaults to None ): The dicriminatory 
    column name. This would be column name along which the graphs would be 
    divided along the subplots. If it is None, it will simply plot a sorted values.
    
    save_png (:obj:`str` ): Whether to save png of results or not. If the value of 
    save_png is not None then it would save the image with name of string value set 
    in save_png.
    
    kwargs: remaining dictionary of keyword arguments for the plot_in function.
    
    """
    
    if dis_col is None:
        simple_plot(save_png, *args, **kwargs)
    else:
        plot_with_discriminator(dis_col, save_png, *args, **kwargs)
