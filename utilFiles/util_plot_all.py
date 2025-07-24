import os
import numpy as np
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")


import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
def plot_functions_1d_np(target_x, target_y, context_x, context_y, pred_y, var, it=0, save_to_dir="", save_img=True):
    plt.rcParams.update({'font.size': 22})

    plt.clf()
    plt.plot(target_x[0], pred_y[0], "b", linewidth=2.5, label = "Model Prediction")
    plt.plot(target_x[0], target_y[0], "k:", linewidth=2, label = "True Function")
    plt.plot(context_x[0], context_y[0], 'ko', markersize=8)
    plt.fill_between(
        target_x[0, :, 0],
        pred_y[0, :, 0] - var[0, :, 0],
        pred_y[0, :, 0] + var[0, :, 0],
        alpha=0.7,
        facecolor='#65c999',
        interpolate=True,
        label = "Variance"
    )
    plt.xlabel("X value")
    plt.ylabel("Y value")
    # plt.xlim(-1.5,1.5)
    # plt.ylim(-1.5,1.5)
    # plt.ylim(-3,7)
    # plt.ylim(-3,5)
    # plt.ylim(-0.7,1.5)

    # plot details
    plt.grid(False)
    plt.legend(loc='upper right')
    # plt.show()
    if save_img:
        plt.savefig(f"{save_to_dir}/plotv5{it}" + ".png", format='png',dpi=300,bbox_inches='tight')
        # plt.show()
    else:
        plt.show()

def plot_functions_alea_ep_1d(target_x, target_y, context_x, context_y, pred_y, epis, alea, save_img=True, save_to_dir="eval_images", save_name="a.png", datetime_data=None):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import pandas as pd
    plt.rcParams.update({'font.size': 16})
    plt.clf()

    # Create distributed x values between 0 and 1
    if target_x.ndim > 2:  # If shape is like [batch, points, features]
        # Take only the first batch
        target_x = target_x[0]  # Shape becomes [points, features]
        target_y = target_y[0]  # Shape becomes [points, 1]
        context_x = context_x[0]  # Shape becomes [points, features]
        context_y = context_y[0]  # Shape becomes [points, 1]
        pred_y = pred_y[0]      # Shape becomes [points, 1]
        epis = epis[0]          # Shape becomes [points, 1]
        
        # Take only the first point from each sequence and flatten
        target_x = target_x[:, 0].flatten()  # Shape becomes [points]
        target_y = target_y[:, 0].flatten()  # Shape becomes [points]
        context_x = context_x[:, 0].flatten()  # Shape becomes [points]
        context_y = context_y[:, 0].flatten()  # Shape becomes [points]
        pred_y = pred_y[:, 0].flatten()      # Shape becomes [points]
        epis = epis[:, 0].flatten()          # Shape becomes [points]
        
        # Use datetime data if provided, otherwise create evenly spaced x values
        if datetime_data is not None and 'target_datetime' in datetime_data:
            target_datetime = datetime_data['target_datetime']
            num_target_points = len(target_y)
            
            # Convert to pandas datetime if not already
            datetime_x = pd.to_datetime(target_datetime)
            
            # Make sure we have the right number of points
            if len(datetime_x) >= num_target_points:
                datetime_x = datetime_x[:num_target_points]
                distributed_x = datetime_x
            else:
                # Fallback to distributed values if not enough datetime points
                distributed_x = np.linspace(0, 1, num_target_points)
                datetime_x = None
        else:
            # Fallback to distributed values
            num_target_points = len(target_y)
            distributed_x = np.linspace(0, 1, num_target_points)
            datetime_x = None
            
    else:
        # If already 2D, ensure they're flattened
        distributed_x = np.asarray(target_x).flatten()
        target_y = np.asarray(target_y).flatten()
        context_y = np.asarray(context_y).flatten()
        pred_y = np.asarray(pred_y).flatten()
        epis = np.asarray(epis).flatten()
        datetime_x = None

    # Use datetime_x if available, otherwise use distributed_x
    x_axis = datetime_x if datetime_x is not None else distributed_x

    # Plot the data
    plt.plot(x_axis, target_y, "k:", linewidth=2.6, label="True Function")
    plt.plot(x_axis, pred_y, "b", linewidth=2, label="Prediction")
    # plt.plot(context_x_scaled, context_y, 'ko', markersize=6, label="Context Points")

    # Plot vertical line (adjust position based on data type)
    if datetime_x is not None:
        # For datetime data, plot vertical line at 85% of the time range
        x_min, x_max = x_axis.min(), x_axis.max()
        x_line = x_min + 0.85 * (x_max - x_min)
        y_min, y_max = min(target_y.min(), pred_y.min()), max(target_y.max(), pred_y.max())
        plt.vlines(x=x_line, ymin=y_min-0.1, ymax=y_max+0.1, linestyles='--', colors='gray')
    else:
        # For normalized data, use original position
        plt.vlines(x=0.85, ymin=-0.2, ymax=1.2, linestyles='--', colors='gray')

    plt.title("ENP-C")

    # Plot epistemic uncertainty
    plt.fill_between(
        x_axis,
        pred_y - epis,
        pred_y + epis,
        alpha=0.7,
        facecolor='#65c999',
        interpolate=True,
        label="Epistemic Unc."
    )
    # plt.fill_between(
    #     target_x[0, :, 0],
    #     pred_y[0, :, 0] - alea[0, :, 0],
    #     pred_y[0, :, 0] + alea[0, :, 0],
    #     alpha=0.2,
    #     facecolor='red',
    #     interpolate=True,
    #     label="Aleatoric"
    # )

    # Set appropriate labels and limits
    if datetime_x is not None:
        plt.xlabel("Date")
        # Format x-axis for datetime
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(datetime_x)//5)))
        plt.xticks(rotation=45)
        # Auto-adjust y-limits based on data
        y_min = min(target_y.min(), pred_y.min())
        y_max = max(target_y.max(), pred_y.max())
        y_margin = 0.1 * (y_max - y_min)
        plt.ylim(y_min - y_margin, y_max + y_margin)
    else:
        plt.xlabel("X value")
        plt.xlim(0, 1)  # Normalized range
        plt.ylim(-0.2, 1.2)  # Add some margin for clarity

    plt.ylabel("Y value")
    plt.grid(True)
    plt.legend(fontsize='small')

    if not os.path.exists(save_to_dir):
        os.mkdir(save_to_dir)
    if save_img:
        plt.savefig(f"{save_to_dir}/Al5Ep{save_name}.png", format='png', dpi=300, bbox_inches='tight')
    else:
        plt.show()

def plot_functions_var_1d(target_x, target_y, context_x, context_y, pred_y, var,save_img=True,save_to_dir="eval_images", save_name="a.png"):
    plt.rcParams.update({'font.size': 22})
    plt.clf()
    plt.plot(target_x[0], target_y[0], "k:", linewidth=2.6, label = "True Function")
    plt.plot(target_x[0], pred_y[0], "b", linewidth=2, label = "Prediction")
    plt.plot(context_x[0], context_y[0], 'ko', markersize=8)
    plt.vlines(x=5.0, ymin=-4, ymax=8, linestyles='--')
    plt.title("CNP Model")
    plt.fill_between(
        target_x[0, :, 0],
        pred_y[0, :, 0] - var[0, :, 0],
        pred_y[0, :, 0] + var[0, :, 0],
        alpha=0.7,
        facecolor='#65c999',
        interpolate=True,
        label = "Variance",
    )
    # plt.fill_between(
    #     target_x[0, :, 0],
    #     pred_y[0, :, 0] - alea[0, :, 0],
    #     pred_y[0, :, 0] + alea[0, :, 0],
    #     alpha=0.2,
    #     facecolor='red',
    #     interpolate=True,
    #     label = "Aleatoric",
    # )

    plt.xlabel("X value")
    plt.ylabel("Y value")
    plt.xlim(-5,10)
    plt.ylim(-5,12)

    # plt.xlim(-2,3)
    # plt.ylim(-3,3)
    # plt.ylim(-0.6,1.0)
    # plt.ylim(-0.7,1.5)

    # plot details
    plt.grid(False)
    # plt.legend(loc=2, bbox_to_anchor=(1,1))
    plt.legend()#loc='upper left')
    # plt.show()
    if not os.path.exists(save_to_dir):
        os.mkdir(save_to_dir)
    if save_img:
        plt.savefig(f"{save_to_dir}/Al5Ep{save_name}" + ".png", format='png',dpi=300,bbox_inches='tight')
        # plt.show()
    else:
        plt.show()


def plot_functions_multiple(target_x, target_y, context_x, context_y, predictions, epis=0,alea=0,save_img=True,save_to_dir="eval_images", save_name="a.png"):
    plt.rcParams.update({'font.size': 22})
    plt.clf()
    plt.plot(target_x[0], target_y[0], "k:", linewidth=1.6, label = "True Function")
    for index, pred_y in enumerate(predictions):
        if index == 0:
            plt.plot(target_x[0], pred_y[0], "b", linewidth=1, label="Predictions")
        else:
            plt.plot(target_x[0], pred_y[0], "b", linewidth=1)
    plt.plot(context_x[0], context_y[0], 'ko', markersize=8)
    print("min: ", min(min(target_y)))
    plt.ylim(( min(min(target_y))-0.5, max(max(target_y))+1.0))
    # plt.vlines(x=5.0, ymin=-5, ymax=5)
    # plt.fill_between(
    #     target_x[0, :, 0],
    #     pred_y[0, :, 0] - epis[0, :, 0],
    #     pred_y[0, :, 0] + epis[0, :, 0],
    #     alpha=0.7,
    #     facecolor='#65c999',
    #     interpolate=True,
    #     label = "Epistemic Uncertainty",
    # )
    # plt.fill_between(
    #     target_x[0, :, 0],
    #     pred_y[0, :, 0] - alea[0, :, 0],
    #     pred_y[0, :, 0] + alea[0, :, 0],
    #     alpha=0.2,
    #     facecolor='red',
    #     interpolate=True,
    #     label = "Aleatoric",
    # )

    plt.xlabel("X value")
    plt.ylabel("Y value")
    # plt.xlim(-1.5,1.5)
    # plt.ylim(-0.75,1.2)
    # plt.ylim(-0.6,1.0)
    # plt.ylim(-3,5)

    # plot details
    plt.grid(False)
    plt.legend(loc="upper right")
    # plt.show()
    if not os.path.exists(save_to_dir):
        os.mkdir(save_to_dir)
    if save_img:
        plt.savefig(f"{save_to_dir}/a{save_name}" + ".png", format='png',dpi=300,bbox_inches='tight')
        # plt.show()
    else:
        plt.show()


def plot_functions_alea_ep_1d_with_original(
    target_x,
    target_y,
    context_x,
    context_y,
    pred_y,
    epistemic,
    aleatoric,
    dataset,  # Pass the dataset object for inverse transformation
    save_img=True,
    save_to_dir="",
    save_name="",
    datetime_data=None,
):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import pandas as pd
    import numpy as np
    
    plt.clf()  # Clear the current figure
    
    # Move tensors to CPU and convert to numpy for plotting
    target_x_cpu = target_x.cpu().detach()
    target_y_cpu = target_y.cpu().detach()
    context_x_cpu = context_x.cpu().detach()
    context_y_cpu = context_y.cpu().detach()
    pred_y_cpu = pred_y.cpu().detach()
    epistemic_cpu = epistemic.cpu().detach()
    aleatoric_cpu = aleatoric.cpu().detach()
    
    # Transform only the y-values (prices) back to original scale
    target_y_orig = dataset.inverse_transform(target_y_cpu[0, :, 0, 0], 'close')
    # context_y_orig = dataset.inverse_transform(context_y_cpu[0, :, 0, 0], 'close')
    pred_y_orig = dataset.inverse_transform(pred_y_cpu[0, :, 0, 0], 'close')
    
    # Determine x-axis values
    if datetime_data is not None and 'target_datetime' in datetime_data:
        target_datetime = datetime_data['target_datetime']
        num_target_points = len(target_y_orig)
        
        # Convert to pandas datetime if not already
        datetime_x = pd.to_datetime(target_datetime)
        
        # Generate evenly distributed datetime points between start and end
        if len(datetime_x) >= 2:  # We need at least start and end dates
            start_date = datetime_x.min()
            end_date = datetime_x.max()
            # Create evenly distributed dates between start and end
            x_points = pd.date_range(start=start_date, end=end_date, periods=num_target_points)
        elif len(datetime_x) >= num_target_points:
            datetime_x = datetime_x[:num_target_points]
            x_points = datetime_x
        else:
            # Fallback to time step indices if not enough datetime points
            x_points = np.arange(len(target_y_orig))
    else:
        # Fallback to time step indices
        x_points = np.arange(len(target_y_orig))
    
    # Plot original scale data
    plt.plot(x_points, target_y_orig, "k:", linewidth=2, label="Target")
    plt.plot(x_points, pred_y_orig, "b", linewidth=2, label="Prediction")
    
    # Transform uncertainties to original scale using bounds approach
    pred_plus_epis = dataset.inverse_transform(pred_y_cpu[0, :, 0, 0] + epistemic_cpu[0, :, 0, 0], 'close')
    pred_minus_epis = dataset.inverse_transform(pred_y_cpu[0, :, 0, 0] - epistemic_cpu[0, :, 0, 0], 'close')
    epistemic_orig = (pred_plus_epis - pred_minus_epis) / 2
    
    pred_plus_alea = dataset.inverse_transform(pred_y_cpu[0, :, 0, 0] + aleatoric_cpu[0, :, 0, 0], 'close')
    pred_minus_alea = dataset.inverse_transform(pred_y_cpu[0, :, 0, 0] - aleatoric_cpu[0, :, 0, 0], 'close')
    aleatoric_orig = (pred_plus_alea - pred_minus_alea) / 2
    
    # Plot epistemic uncertainty
    plt.fill_between(
        x_points,
        pred_y_orig - epistemic_orig,
        pred_y_orig + epistemic_orig,
        alpha=0.7,
        facecolor='#65c999',
        interpolate=True,
        label="Epistemic Unc."
    )
    
    # Plot aleatoric uncertainty
    plt.fill_between(
        x_points,
        pred_y_orig - aleatoric_orig,
        pred_y_orig + aleatoric_orig,
        alpha=0.3,
        facecolor='red',
        interpolate=True,
        label="Aleatoric Unc."
    )
    
    # Check if we're using datetime data
    is_datetime = isinstance(x_points, pd.DatetimeIndex) or (hasattr(x_points, 'dtype') and 'datetime' in str(x_points.dtype))
    
    # Plot vertical line
    if is_datetime:
        # For datetime data, plot vertical line at 85% of the time range
        x_min, x_max = x_points.min(), x_points.max()
        x_line = x_min + 0.85 * (x_max - x_min)
        plt.vlines(x=x_line, ymin=target_y_orig.min(), ymax=target_y_orig.max(), linestyles='--', colors='gray')
    else:
        # For time step data, use original position
        plt.vlines(x=310, ymin=target_y_orig.min(), ymax=target_y_orig.max(), linestyles='--', colors='gray')
    
    print('target_y_orig[-10]', target_y_orig[-10])
    print('pred_y_orig[-10]', pred_y_orig[-10])
    
    plt.title("Original Values (XAUUSD Price)")
    plt.legend(fontsize='small')
    plt.grid(True)
    plt.ylabel("Price (USD)")
    
    # Set appropriate x-axis label and formatting
    if is_datetime:
        plt.xlabel("Date")
        
        # Calculate the time span to choose appropriate formatting
        time_span = (x_points.max() - x_points.min()).total_seconds()
        time_span_days = time_span / (24 * 3600)  # Convert to days
        
        if time_span_days <= 1:  # Less than 1 day - show hours
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=max(1, int(time_span_days * 24 / 5))))
        elif time_span_days <= 7:  # Less than 1 week - show days with time
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=max(12, int(time_span_days * 24 / 5))))
        elif time_span_days <= 30:  # Less than 1 month - show days
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=max(1, int(time_span_days / 5))))
        else:  # More than 1 month - show dates
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=max(1, int(time_span_days / 5))))
            
        plt.xticks(rotation=45)
    else:
        plt.xlabel("Time Step")
    
    if not os.path.exists(save_to_dir):
        os.makedirs(save_to_dir)
        
    if save_img:
        plt.savefig(f"{save_to_dir}/{save_name}_original.png", format='png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()