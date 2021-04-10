# visualization
import datetime
import matplotlib.pyplot as plt

def heatmap2d(data, valueL='lastrow', color='cool', size=12, marker='o',alpha=0.4,
                save=False, savepath='./' ):
    """The correlation of each column and the magnitude of the evaluation value are displayed in color.
    Enter the data you want to see in data and the evaluation value in valueL. If not entered, the last column of data"""
    from pandas.plotting import scatter_matrix
    data = np.array(data)
    if valueL=='lastrow':
        valueL=data[:, -1]
        data = np.delete(data, obj=-1, axis=1)

    normalizedValueL = list( (valueL - min(valueL)) / (max(valueL) - min(valueL)) )

    if color=='hot':
        colors = plt.cm.hot_r(normalizedValueL)
        # For color bar display
        colmap = plt.cm.ScalarMappable(cmap=plt.cm.hot_r)
    elif color=='cool':
        colors = plt.cm.cool_r(normalizedValueL)
        colmap = plt.cm.ScalarMappable(cmap=plt.cm.cool_r)
    elif color=='hsv':
        colors = plt.cm.hsv_r(normalizedValueL)
        colmap = plt.cm.ScalarMappable(cmap=plt.cm.hsv_r)
    elif color=='jet':
        colors = plt.cm.jet_r(normalizedValueL)
        colmap = plt.cm.ScalarMappable(cmap=plt.cm.jet_r)
    elif color=='gray':
        colors = plt.cm.gray_r(normalizedValueL)
        colmap = plt.cm.ScalarMappable(cmap=plt.cm.gray_r)
    elif color=='spring':
        colors = plt.cm.spring_r(normalizedValueL)
        colmap = plt.cm.ScalarMappable(cmap=plt.cm.spring_r)
    elif color=='summer':
        colors = plt.cm.summer_r(normalizedValueL)
        colmap = plt.cm.ScalarMappable(cmap=plt.cm.summer_r)
    elif color=='autumn':
        colors = plt.cm.autumn_r(normalizedValueL)
        colmap = plt.cm.ScalarMappable(cmap=plt.cm.autumn_r)
    elif color=='winter':
        colors = plt.cm.winter_r(normalizedValueL)
        colmap = plt.cm.ScalarMappable(cmap=plt.cm.winter_r)
    else:
        print('Since there is no color, it will be the default cool')
        colors = plt.cm.cool_r(normalizedValueL)
        colmap = plt.cm.ScalarMappable(cmap=plt.cm.cool_r)

    colmap.set_array(valueL)

    plt.figure()

    ax_matrix = scatter_matrix(pd.DataFrame(data), c=colors, s=size, marker=marker, alpha=alpha)

    # For color bar display
    plt.colorbar(colmap, ax=ax_matrix)
    if save==True:
        date = datetime.datetime.now()
        plt.savefig(savepath+'scatter_matrix_'+str(date.year)+'_'+ str(date.month)+ \
                    '_'+str(date.day)+'_'+str(date.hour)+'_'+ \
                    str(date.minute)+'_'+str(date.second), dpi=150)
    plt.show()



def heatmap3d(xL, yL ,zL, valueL, grid=True, color='cool',
                size=100, marker='o',alpha=0.8,save=False, savepath='./'):
    """Look at the evaluation of values in three dimensions. The value is expressed in color."""
    from mpl_toolkits.mplot3d import Axes3D
    #Normalize valueL into 0 to 1
    normalizedValueL = list( (valueL - min(valueL)) / (max(valueL) - min(valueL)) )

    if color=='hot':
        colors = plt.cm.hot_r(normalizedValueL)
        # For color bar display
        colmap = plt.cm.ScalarMappable(cmap=plt.cm.hot_r)
    elif color=='cool':
        colors = plt.cm.cool_r(normalizedValueL)
        colmap = plt.cm.ScalarMappable(cmap=plt.cm.cool_r)
    elif color=='hsv':
        colors = plt.cm.hsv_r(normalizedValueL)
        colmap = plt.cm.ScalarMappable(cmap=plt.cm.hsv_r)
    elif color=='jet':
        colors = plt.cm.jet_r(normalizedValueL)
        colmap = plt.cm.ScalarMappable(cmap=plt.cm.jet_r)
    elif color=='gray':
        colors = plt.cm.gray_r(normalizedValueL)
        colmap = plt.cm.ScalarMappable(cmap=plt.cm.gray_r)
    elif color=='spring':
        colors = plt.cm.spring_r(normalizedValueL)
        colmap = plt.cm.ScalarMappable(cmap=plt.cm.spring_r)
    elif color=='summer':
        colors = plt.cm.summer_r(normalizedValueL)
        colmap = plt.cm.ScalarMappable(cmap=plt.cm.summer_r)
    elif color=='autumn':
        colors = plt.cm.autumn_r(normalizedValueL)
        colmap = plt.cm.ScalarMappable(cmap=plt.cm.autumn_r)
    elif color=='winter':
        colors = plt.cm.winter_r(normalizedValueL)
        colmap = plt.cm.ScalarMappable(cmap=plt.cm.winter_r)
    else:
        print('Since there is no color, it will be the default cool')
        colors = plt.cm.cool_r(normalizedValueL)
        colmap = plt.cm.ScalarMappable(cmap=plt.cm.cool_r)

    colmap.set_array(valueL)

    fig = plt.figure()
    ax = Axes3D(fig)

    # Set the grid on of off
    if not grid:
        ax.grid(False)

    ax.scatter(xL,yL,zL, s =size, c=colors, marker=marker, alpha=alpha)
    # For color bar display
    cb = fig.colorbar(colmap)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if save==True:
        date = datetime.datetime.now()
        plt.savefig(savepath+'3Dheatmap_'+str(date.year)+'_'+ str(date.month)+ \
                   '_'+str(date.day)+'_'+str(date.hour)+'_'+ \
                   str(date.minute)+'_'+str(date.second), dpi=150)
    plt.show()


def heatmap3d_plotly(xL, yL , zL, valueL, size=6, alpha=0.8, color='Viridis'):
    '''3D heat map plotly version'''
    import plotly.offline as offline
    import plotly.graph_objs as go
    # There is no setting for the storage method, so consider it
    trace1 = go.Scatter3d(
    x=xL,
    y=yL,
    z=zL,
    mode='markers',
    marker=dict(
        size=size,
        color=valueL,                # set color to an array/list of desired values
        colorscale=color,   # choose a colorscale
        opacity=alpha,
        colorbar=dict(thickness=20)
        )
    )

    data = [trace1]
    layout = go.Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )
    fig = go.Figure(data=data, layout=layout)
    offline.plot(fig, filename='3d-scatter-colorscale')