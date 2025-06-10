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

def plot_functions_alea_ep_1d(target_x, target_y, context_x, context_y, pred_y, epis, alea, save_img=True, save_to_dir="eval_images", save_name="a.png"):
    import matplotlib.pyplot as plt
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
        
        # Create evenly spaced x values for target points
        num_target_points = len(target_x)
        distributed_x = np.linspace(0, 1, num_target_points)
        
    else:
        # If already 2D, ensure they're flattened
        distributed_x = np.asarray(target_x).flatten()
        target_y = np.asarray(target_y).flatten()
        context_y = np.asarray(context_y).flatten()
        pred_y = np.asarray(pred_y).flatten()
        epis = np.asarray(epis).flatten()

    # Plot the data
    plt.plot(distributed_x, target_y, "k:", linewidth=2.6, label="True Function")
    plt.plot(distributed_x, pred_y, "b", linewidth=2, label="Prediction")
    # plt.plot(context_x_scaled, context_y, 'ko', markersize=6, label="Context Points")

    # Plot vertical line at x=0.85
    plt.vlines(x=0.85, ymin=-0.2, ymax=1.2, linestyles='--', colors='gray')

    plt.title("ENP-C")

    # Plot epistemic uncertainty
    plt.fill_between(
        distributed_x,
        pred_y - epis,
        pred_y + epis,
        alpha=0.7,
        facecolor='#65c999',
        interpolate=True,
        label="Epistemic Unc."
    )



    # plt.plot(target_x[0], target_y[0], "k:", linewidth=2.6, label="True Function")
    # plt.plot(target_x[0], pred_y[0], "b", linewidth=2, label="Prediction")
    # plt.plot(context_x[0], context_y[0], 'ko', markersize=6)  # اندازه نشانگرها کاهش یافت
    # plt.vlines(x=0.5, ymin=-0.2, ymax=1.2, linestyles='--')  # خط عمودی به x=0.5 تغییر کرد
    # plt.title(r"ENP-C")
    # plt.fill_between(
    #     target_x[0, :, 0],
    #     pred_y[0, :, 0] - epis[0, :, 0],
    #     pred_y[0, :, 0] + epis[0, :, 0],
    #     alpha=0.7,
    #     facecolor='#65c999',
    #     interpolate=True,
    #     label="Epistemic Unc."
    # )
    # plt.fill_between(
    #     target_x[0, :, 0],
    #     pred_y[0, :, 0] - alea[0, :, 0],
    #     pred_y[0, :, 0] + alea[0, :, 0],
    #     alpha=0.2,
    #     facecolor='red',
    #     interpolate=True,
    #     label="Aleatoric"
    # )

    plt.xlabel("X value")
    plt.ylabel("Y value")
    plt.xlim(0, 1)  # Normalized range
    plt.ylim(-0.2, 1.2)  # Add some margin for clarity

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
):
    import matplotlib.pyplot as plt
    import numpy as np
    
    plt.clf()  # Clear the current figure
    
    # Create time points for x-axis
    x_points = np.arange(len(target_x[0, :, 0, 0]))
    
    # Move tensors to CPU and convert to numpy for plotting
    target_x_cpu = target_x.cpu().detach()
    target_y_cpu = target_y.cpu().detach()
    context_x_cpu = context_x.cpu().detach()
    context_y_cpu = context_y.cpu().detach()
    pred_y_cpu = pred_y.cpu().detach()
    epistemic_cpu = epistemic.cpu().detach()
    
    # Transform only the y-values (prices) back to original scale
    target_y_orig = dataset.inverse_transform(target_y_cpu[0, :, 0, 0], 'close')
    context_y_orig = dataset.inverse_transform(context_y_cpu[0, :, 0, 0], 'close')
    pred_y_orig = dataset.inverse_transform(pred_y_cpu[0, :, 0, 0], 'close')
    
    # Plot original scale data
    plt.plot(x_points, target_y_orig, "k:", linewidth=2, label="Target")
    plt.plot(x_points, pred_y_orig, "b", linewidth=2, label="Prediction")
    
    # Plot vertical line at x=340
    plt.vlines(x=300, ymin=target_y_orig.min(), ymax=target_y_orig.max(), linestyles='--', colors='gray')

    
    plt.title("Original Values (XAUUSD Price)")
    plt.legend(fontsize='small')
    plt.grid(True)
    plt.ylabel("Price (USD)")
    plt.xlabel("Time Step")
    
    if not os.path.exists(save_to_dir):
        os.makedirs(save_to_dir)
        
    if save_img:
        plt.savefig(f"{save_to_dir}/{save_name}_original.png", format='png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()