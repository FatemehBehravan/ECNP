import torch
import numpy as np

def create_k_fold_indices(num_points, k=5, shuffle=True, random_state=42):
    """
    Create k-fold cross-validation indices without sklearn dependency
    
    Args:
        num_points: Number of data points
        k: Number of folds
        shuffle: Whether to shuffle indices
        random_state: Random seed for reproducibility
    
    Returns:
        List of tuples (train_indices, val_indices) for each fold
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    indices = np.arange(num_points)
    if shuffle:
        np.random.shuffle(indices)
    
    fold_sizes = np.full(k, num_points // k, dtype=int)
    fold_sizes[:num_points % k] += 1
    
    folds = []
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        folds.append(indices[start:stop])
        current = stop
    
    # Create train/val splits for each fold
    splits = []
    for i in range(k):
        val_indices = folds[i]
        train_indices = np.concatenate([folds[j] for j in range(k) if j != i])
        splits.append((train_indices, val_indices))
    
    return splits


def add_to_logging_dict(logging_dict, header, values):
    for index, a in enumerate(values):
        a_mean = torch.mean(a).detach().cpu().numpy()
        a_variance = torch.var(a).detach().cpu().numpy()
        # print(names[index], " " , a_mean, a_variance)
        logging_dict[header[index]+"_mean"] = a_mean
        # logging_dict[header[index]+"_variance"] = a_variance
    return logging_dict


def NIG_NLL(it, y, mu, v, alpha, beta):
    epsilon = 1e-16
    twoBlambda = 2*beta*(1+v)

    a1 = 0.5723649429247001 - 0.5 * torch.log(v+epsilon)
    # a1 = 0.5 * torch.log(np.pi / torch.max(v,epsilon))
    a2a = - alpha*torch.log( 2*beta +epsilon)
    a2b = - alpha * torch.log(1 + v)
    a3 = (alpha+0.5) * torch.log( v*(y-mu)**2 + twoBlambda + epsilon)
    a4 = torch.lgamma(alpha) - torch.lgamma(alpha+0.5)

    a2 = a2a + a2b

    nll = a1 + a2 + a3 + a4

    # nll = torch.exp(nll)
    likelihood = (np.pi/ v)**(0.5) / (twoBlambda**alpha)  * ((v*(y-mu)**2 + twoBlambda)**(alpha+0.5))
    # nll = 1 * (y - mu)**2
    likelihood *= torch.exp (a4)
    # nll = likelihood


    mse = (mu - y)**2
    mse += 1e-15
    mse = torch.log(mse) 

    header = ['y','mu', 'v', 'alpha', 'beta', 'nll', 'mse', 'a1', 'a2a','a2b','a2', 'a3', 'a4', 'likelihood', 'twoblambda']
    values = [y, mu, v, alpha, beta, nll, mse, a1, a2a,a2b,a2, a3, a4, likelihood, twoBlambda]

    logging_dict = {}
    logging_dict['Iteration']= it
    logging_dict = add_to_logging_dict(logging_dict, header, values)

    is_nan = torch.stack([torch.isnan(x)*1 for x in values])

    return nll, logging_dict

def NIG_Reg(y, gamma, v, alpha, beta):

    error = torch.abs(y-gamma)

    # alternatively can do
    # error = (y-gamma)**2

    evi = v + alpha + 1/(beta+1e-15)
    reg = error*evi

    return reg
    # return torch.mean(reg)


def calculate_evidential_loss(it, y, mu, v, alpha, beta, lambda_coef=1.0):

    nig_nll, logging_dict = NIG_NLL(it, y, mu, v, alpha,beta)
    nig_reg =NIG_Reg(y, mu, v, alpha,beta)

    ev_sum = nig_nll  + lambda_coef*nig_reg
    evidential_loss = torch.mean(ev_sum)

    header = ['nig_nll', 'nig_reg', 'nig_loss', 'mse']
    values = [nig_nll, nig_reg, ev_sum, (y-mu)**2]
    logging_dict = add_to_logging_dict(logging_dict, header, values)

    return evidential_loss, logging_dict

def calculate_evidential_loss_constraints(it, y, mu, v, alpha, beta, lambda_coef=1.0):

    nig_nll, logging_dict = NIG_NLL(it, y, mu, v, alpha,beta)
    nig_reg =NIG_Reg(y, mu, v, alpha,beta)

    ev_sum = nig_nll  + lambda_coef*nig_reg
    evidential_loss = torch.mean(ev_sum)

    header = ['nig_nll', 'nig_reg', 'nig_loss', 'mse']
    values = [nig_nll, nig_reg, ev_sum, (y-mu)**2]
    logging_dict = add_to_logging_dict(logging_dict, header, values)

    return evidential_loss, logging_dict

def calc_ev_krnl_reg(target_x, context_x,v, lambda_ker = 0):

    diff_mat = target_x[:,:,None,:] - context_x[:,None,:,:]
    sq_mat = diff_mat**2

    dist_mat = torch.einsum('bijk->bij', sq_mat)
    dist_mat = torch.sqrt(dist_mat)

    min_dist = torch.min(dist_mat, dim = -1)[0]

    kernel_reg_val = lambda_ker * min_dist[:,:,None]*v

    kernel_reg_val = torch.mean(kernel_reg_val)

    return kernel_reg_val


def calculate_cross_validation_loss(model, query, target_y, k=5, lambda_coef=1.0, device='cuda'):
    """
    Calculate k-fold cross-validation loss with k=5
    
    Args:
        model: The neural process model
        query: Input query points [batch_size, num_points, input_dim]
        target_y: Target values [batch_size, num_points, ...] (supports multiple dimensions)
        k: Number of folds for cross-validation (default=5)
        lambda_coef: Regularization coefficient for evidential loss
        device: Device to run computations on
    
    Returns:
        cv_loss: Average cross-validation loss across all folds
        cv_logging_dict: Dictionary containing detailed metrics from each fold
    """
    
    # Handle different tensor shapes
    if len(target_y.shape) == 3:
        batch_size, num_points, output_dim = target_y.shape
    elif len(target_y.shape) == 4:
        batch_size, num_points, dim_3, dim_4 = target_y.shape
        # Reshape to 3D for cross-validation
        target_y = target_y.view(batch_size, num_points, -1)
        output_dim = dim_3 * dim_4
    else:
        raise ValueError(f"Unsupported tensor shape: {target_y.shape}. Expected 3D or 4D tensor.")
    
    # Initialize logging dictionary
    cv_logging_dict = {
        'fold_losses': [],
        'fold_nll': [],
        'fold_reg': [],
        'fold_mse': []
    }
    
    total_cv_loss = 0.0
    
    for batch_idx in range(batch_size):
        batch_query = query[batch_idx:batch_idx+1]  # [1, num_points, input_dim]
        batch_target = target_y[batch_idx:batch_idx+1]  # [1, num_points, output_dim]
        
        # Create k-fold splits for current batch
        kfold_splits = create_k_fold_indices(num_points, k=k, shuffle=True, random_state=42)
        
        batch_cv_loss = 0.0
        fold_count = 0
        
        for train_idx, val_idx in kfold_splits:
            # Split data into training and validation for this fold
            train_query = batch_query[:, train_idx, :].to(device)
            train_target = batch_target[:, train_idx, :].to(device)
            val_query = batch_query[:, val_idx, :].to(device)
            val_target = batch_target[:, val_idx, :].to(device)
            
            # Create combined query for the model (train + val)
            combined_query = torch.cat([train_query, val_query], dim=1)
            combined_target = torch.cat([train_target, val_target], dim=1)
            
            model.eval()
            with torch.no_grad():
                # Forward pass through model
                if hasattr(model, 'forward') and len(model.forward.__code__.co_varnames) >= 3:
                    # For models that take iteration parameter
                    _, recons_loss, kl_loss, loss, mu, v, alpha, beta = model(combined_query, combined_target, fold_count)
                else:
                    # For models without iteration parameter
                    outputs = model(combined_query, combined_target)
                    if len(outputs) == 8:
                        _, recons_loss, kl_loss, loss, mu, v, alpha, beta = outputs
                    else:
                        # Handle different model output formats
                        mu, v, alpha, beta = outputs[-4:]
                
                # Extract validation predictions
                val_mu = mu[:, len(train_idx):, :]
                val_v = v[:, len(train_idx):, :]
                val_alpha = alpha[:, len(train_idx):, :]
                val_beta = beta[:, len(train_idx):, :]
                
                # Calculate evidential loss on validation set
                fold_loss, fold_logging = calculate_evidential_loss(
                    fold_count, val_target, val_mu, val_v, val_alpha, val_beta, lambda_coef
                )
                
                batch_cv_loss += fold_loss.item()
                fold_count += 1
                
                # Store fold metrics
                cv_logging_dict['fold_losses'].append(fold_loss.item())
                cv_logging_dict['fold_nll'].append(fold_logging.get('nig_nll_mean', 0))
                cv_logging_dict['fold_reg'].append(fold_logging.get('nig_reg_mean', 0))
                cv_logging_dict['fold_mse'].append(fold_logging.get('mse_mean', 0))
        
        # Average loss across folds for this batch
        batch_cv_loss /= k
        total_cv_loss += batch_cv_loss
    
    # Average loss across all batches
    cv_loss = total_cv_loss / batch_size
    
    # Calculate summary statistics
    cv_logging_dict['mean_cv_loss'] = cv_loss
    cv_logging_dict['std_fold_loss'] = np.std(cv_logging_dict['fold_losses'])
    cv_logging_dict['mean_fold_nll'] = np.mean(cv_logging_dict['fold_nll'])
    cv_logging_dict['mean_fold_reg'] = np.mean(cv_logging_dict['fold_reg'])
    cv_logging_dict['mean_fold_mse'] = np.mean(cv_logging_dict['fold_mse'])
    
    return cv_loss, cv_logging_dict


def calculate_cross_validation_loss_simple(y, mu, v, alpha, beta, k=5, lambda_coef=1.0):
    """
    Simplified k-fold cross-validation loss calculation for pre-computed model outputs
    
    Args:
        y: Target values [batch_size, num_points, ...] (supports multiple dimensions)
        mu: Predicted means [batch_size, num_points, ...]
        v: Predicted variance parameters [batch_size, num_points, ...]
        alpha: Predicted alpha parameters [batch_size, num_points, ...]
        beta: Predicted beta parameters [batch_size, num_points, ...]
        k: Number of folds for cross-validation (default=5)
        lambda_coef: Regularization coefficient for evidential loss
    
    Returns:
        cv_loss: Average cross-validation loss
        cv_logging_dict: Dictionary containing metrics from each fold
    """
    
    # Handle different tensor shapes
    if len(y.shape) == 3:
        batch_size, num_points, output_dim = y.shape
    elif len(y.shape) == 4:
        batch_size, num_points, dim_3, dim_4 = y.shape
        # Reshape to 3D for cross-validation
        y = y.view(batch_size, num_points, -1)
        mu = mu.view(batch_size, num_points, -1)
        v = v.view(batch_size, num_points, -1)
        alpha = alpha.view(batch_size, num_points, -1)
        beta = beta.view(batch_size, num_points, -1)
        output_dim = dim_3 * dim_4
    else:
        raise ValueError(f"Unsupported tensor shape: {y.shape}. Expected 3D or 4D tensor.")
    
    # Initialize logging
    cv_logging_dict = {
        'fold_losses': [],
        'fold_nll': [],
        'fold_reg': [],
        'fold_mse': []
    }
    
    total_cv_loss = 0.0
    
    for batch_idx in range(batch_size):
        # Extract single batch
        batch_y = y[batch_idx:batch_idx+1]
        batch_mu = mu[batch_idx:batch_idx+1]
        batch_v = v[batch_idx:batch_idx+1]
        batch_alpha = alpha[batch_idx:batch_idx+1]
        batch_beta = beta[batch_idx:batch_idx+1]
        
        # Create k-fold splits
        kfold_splits = create_k_fold_indices(num_points, k=k, shuffle=True, random_state=42)
        
        batch_cv_loss = 0.0
        fold_count = 0
        
        for train_idx, val_idx in kfold_splits:
            # Use validation indices for loss calculation
            val_y = batch_y[:, val_idx, :]
            val_mu = batch_mu[:, val_idx, :]
            val_v = batch_v[:, val_idx, :]
            val_alpha = batch_alpha[:, val_idx, :]
            val_beta = batch_beta[:, val_idx, :]
            
            # Calculate evidential loss on validation fold
            fold_loss, fold_logging = calculate_evidential_loss(
                fold_count, val_y, val_mu, val_v, val_alpha, val_beta, lambda_coef
            )
            
            batch_cv_loss += fold_loss.item()
            fold_count += 1
            
            # Store fold metrics
            cv_logging_dict['fold_losses'].append(fold_loss.item())
            cv_logging_dict['fold_nll'].append(fold_logging.get('nig_nll_mean', 0))
            cv_logging_dict['fold_reg'].append(fold_logging.get('nig_reg_mean', 0))
            cv_logging_dict['fold_mse'].append(fold_logging.get('mse_mean', 0))
        
        # Average across folds for this batch
        batch_cv_loss /= k
        total_cv_loss += batch_cv_loss
    
    # Average across batches
    cv_loss = total_cv_loss / batch_size
    
    # Summary statistics
    cv_logging_dict['mean_cv_loss'] = cv_loss
    cv_logging_dict['std_fold_loss'] = np.std(cv_logging_dict['fold_losses'])
    cv_logging_dict['mean_fold_nll'] = np.mean(cv_logging_dict['fold_nll'])
    cv_logging_dict['mean_fold_reg'] = np.mean(cv_logging_dict['fold_reg'])
    cv_logging_dict['mean_fold_mse'] = np.mean(cv_logging_dict['fold_mse'])
    
    return cv_loss, cv_logging_dict



def calculate_r2_score(y, mu, v, alpha, beta):
    """
    Calculate R² (coefficient of determination) for evidential neural processes
    
    R² = 1 - SS_res / SS_tot
    where SS_res = sum of squared residuals
    and SS_tot = total sum of squares
    
    Args:
        y: True values [batch_size, num_points, ...]
        mu: Predicted means [batch_size, num_points, ...]
        v: Predicted variance parameters [batch_size, num_points, ...] (not used for R²)
        alpha: Predicted alpha parameters [batch_size, num_points, ...] (not used for R²)
        beta: Predicted beta parameters [batch_size, num_points, ...] (not used for R²)
    
    Returns:
        r2: R² score
    """
    # Flatten tensors for calculation
    y_flat = y.view(-1)
    mu_flat = mu.view(-1)
    
    # Calculate mean of true values
    y_mean = torch.mean(y_flat)
    
    # Calculate sum of squared residuals (SS_res)
    ss_res = torch.sum((y_flat - mu_flat) ** 2)
    
    # Calculate total sum of squares (SS_tot)
    ss_tot = torch.sum((y_flat - y_mean) ** 2)
    
    # Calculate R²
    r2 = 1 - (ss_res / (ss_tot + 1e-8))  # Add small epsilon to avoid division by zero
    
    return r2


def calculate_evidential_metrics(y, mu, v, alpha, beta):
    """
    Calculate comprehensive evidential metrics including EMSE and R²
    
    Args:
        y: True values [batch_size, num_points, ...]
        mu: Predicted means [batch_size, num_points, ...]
        v: Predicted variance parameters [batch_size, num_points, ...]
        alpha: Predicted alpha parameters [batch_size, num_points, ...]
        beta: Predicted beta parameters [batch_size, num_points, ...]
    
    Returns:
        metrics_dict: Dictionary containing all metrics
    """
    
    # Calculate R²
    r2 = calculate_r2_score(y, mu, v, alpha, beta)
    
    # Calculate traditional MSE
    mse = torch.mean((y - mu) ** 2)
    
    # Calculate RMSE (Root Mean Square Error)
    rmse = torch.sqrt(mse)
    
    # Calculate MAE (Mean Absolute Error)
    mae = torch.mean(torch.abs(y - mu))
    
    # Calculate RMAE (Root Mean Absolute Error)
    rmae = torch.sqrt(mae)
    
    # Calculate epistemic and aleatoric uncertainty
    epis = torch.mean(beta / (v * (alpha - 1)))
    alea = torch.mean(beta / (alpha - 1))
    
    metrics_dict = {
        'r2': r2.item(),
        'mse': mse.item(),
        'rmse': rmse.item(),
        'mae': mae.item(),
        'rmae': rmae.item(),
        'epistemic': epis.item(),
        'aleatoric': alea.item()
    }
    
    return metrics_dict