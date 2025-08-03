#!/usr/bin/env python3
"""
Technical Indicators Analysis Script
Performs correlation analysis and PCA on technical indicators for XAUUSD data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Import our data loading functions
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data.xauusd_data import calculate_technical_indicators

def load_and_prepare_data():
    """Load and prepare the XAUUSD data with technical indicators"""
    print("Loading and preparing XAUUSD data...")
    
    # Load raw data
    file_path = './datasets/Strategy_XAUUSD.csv'
    df = pd.read_csv(file_path, nrows=501)
    
    # Add time features
    hours = pd.to_datetime(df['time'], unit='s').dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * hours / 24)
    df['datetime'] = pd.to_datetime(df['time'], unit='s')
    
    # Calculate technical indicators
    df = calculate_technical_indicators(df)
    
    # Define our feature set (same as in the model)
    feature_columns = [
        'rsi', 'macd', 'macd_signal', 'bb_position',
        'stoch_k', 'momentum_5'
    ]
    
    # Calculate price changes for additional analysis
    df['price_change'] = df['close'].pct_change()
    df['price_change_abs'] = df['close'].diff()
    
    # Remove any NaN values
    df = df.dropna()
    
    return df, feature_columns

def calculate_correlations(df, feature_columns):
    """Calculate Pearson and Spearman correlations"""
    print("Calculating correlations...")
    
    # Prepare data for correlation analysis
    analysis_data = df[feature_columns + ['close', 'price_change', 'price_change_abs']].copy()
    
    # Calculate Pearson correlations
    pearson_corr = analysis_data.corr(method='pearson')
    
    # Calculate Spearman correlations
    spearman_corr = analysis_data.corr(method='spearman')
    
    return pearson_corr, spearman_corr, analysis_data

def create_correlation_heatmaps(pearson_corr, spearman_corr, feature_columns):
    """Create correlation heatmaps"""
    print("Creating correlation heatmaps...")
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # Pearson correlation with close price
    price_corr_pearson = pearson_corr['close'][feature_columns].sort_values(key=abs, ascending=False)
    axes[0, 0].barh(price_corr_pearson.index, price_corr_pearson.values)
    axes[0, 0].set_title('Pearson Correlation with Close Price', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Correlation Coefficient')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add correlation values as text
    for i, v in enumerate(price_corr_pearson.values):
        axes[0, 0].text(v + 0.01 if v >= 0 else v - 0.01, i, f'{v:.3f}', 
                       va='center', ha='left' if v >= 0 else 'right', fontweight='bold')
    
    # Spearman correlation with close price
    price_corr_spearman = spearman_corr['close'][feature_columns].sort_values(key=abs, ascending=False)
    axes[0, 1].barh(price_corr_spearman.index, price_corr_spearman.values, color='orange')
    axes[0, 1].set_title('Spearman Correlation with Close Price', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Correlation Coefficient')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add correlation values as text
    for i, v in enumerate(price_corr_spearman.values):
        axes[0, 1].text(v + 0.01 if v >= 0 else v - 0.01, i, f'{v:.3f}', 
                       va='center', ha='left' if v >= 0 else 'right', fontweight='bold')
    
    # Pearson correlation matrix (features only)
    mask = np.triu(np.ones_like(pearson_corr.loc[feature_columns, feature_columns], dtype=bool))
    sns.heatmap(pearson_corr.loc[feature_columns, feature_columns], 
                mask=mask, annot=True, cmap='RdBu_r', center=0, ax=axes[1, 0],
                square=True, fmt='.2f', cbar_kws={"shrink": .8})
    axes[1, 0].set_title('Pearson Correlation Matrix (Indicators)', fontsize=14, fontweight='bold')
    
    # Spearman correlation matrix (features only)
    sns.heatmap(spearman_corr.loc[feature_columns, feature_columns], 
                mask=mask, annot=True, cmap='RdBu_r', center=0, ax=axes[1, 1],
                square=True, fmt='.2f', cbar_kws={"shrink": .8})
    axes[1, 1].set_title('Spearman Correlation Matrix (Indicators)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('correlation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return price_corr_pearson, price_corr_spearman

def perform_pca_analysis(analysis_data, feature_columns):
    """Perform PCA analysis on the technical indicators"""
    print("Performing PCA analysis...")
    
    # Prepare data for PCA
    X = analysis_data[feature_columns].values
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # Calculate cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    return pca, X_scaled, cumulative_variance, scaler

def create_pca_visualizations(pca, feature_columns, cumulative_variance):
    """Create PCA visualizations"""
    print("Creating PCA visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # Scree Plot
    axes[0, 0].plot(range(1, len(pca.explained_variance_ratio_) + 1), 
                    pca.explained_variance_ratio_, 'bo-', linewidth=2, markersize=8)
    axes[0, 0].set_title('Scree Plot - Explained Variance by Component', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Principal Component')
    axes[0, 0].set_ylabel('Explained Variance Ratio')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add variance percentages as text
    for i, var in enumerate(pca.explained_variance_ratio_):
        axes[0, 0].text(i + 1, var + 0.01, f'{var:.1%}', 
                       ha='center', va='bottom', fontweight='bold')
    
    # Cumulative Explained Variance
    axes[0, 1].plot(range(1, len(cumulative_variance) + 1), 
                    cumulative_variance, 'ro-', linewidth=2, markersize=8)
    axes[0, 1].axhline(y=0.8, color='g', linestyle='--', alpha=0.7, label='80% Variance')
    axes[0, 1].axhline(y=0.9, color='orange', linestyle='--', alpha=0.7, label='90% Variance')
    axes[0, 1].axhline(y=0.95, color='purple', linestyle='--', alpha=0.7, label='95% Variance')
    axes[0, 1].set_title('Cumulative Explained Variance', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Number of Components')
    axes[0, 1].set_ylabel('Cumulative Explained Variance Ratio')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Feature contributions to first principal component
    pc1_contributions = pca.components_[0]
    pc1_abs = np.abs(pc1_contributions)
    sorted_indices = np.argsort(pc1_abs)[::-1]
    
    axes[1, 0].barh([feature_columns[i] for i in sorted_indices], 
                    [pc1_contributions[i] for i in sorted_indices])
    axes[1, 0].set_title('Feature Contributions to PC1', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Contribution Weight')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Feature contributions to second principal component
    pc2_contributions = pca.components_[1]
    pc2_abs = np.abs(pc2_contributions)
    sorted_indices_pc2 = np.argsort(pc2_abs)[::-1]
    
    axes[1, 1].barh([feature_columns[i] for i in sorted_indices_pc2], 
                    [pc2_contributions[i] for i in sorted_indices_pc2], color='orange')
    axes[1, 1].set_title('Feature Contributions to PC2', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Contribution Weight')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pca_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return pc1_contributions, pc2_contributions

def create_feature_importance_analysis(pca, feature_columns, price_corr_pearson):
    """Create a comprehensive feature importance analysis"""
    print("Creating feature importance analysis...")
    
    # Calculate feature importance based on PCA loadings
    # Take absolute values and weight by explained variance
    feature_importance = np.zeros(len(feature_columns))
    
    # Weight by first 3 components (usually capture most variance)
    for i in range(min(3, len(pca.components_))):
        feature_importance += np.abs(pca.components_[i]) * pca.explained_variance_ratio_[i]
    
    # Normalize to 0-1 scale
    feature_importance = feature_importance / np.max(feature_importance)
    
    # Create comprehensive importance dataframe
    importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'PCA_Importance': feature_importance,
        'Price_Correlation_Abs': np.abs(price_corr_pearson[feature_columns].values),
        'PC1_Contribution_Abs': np.abs(pca.components_[0]),
        'PC2_Contribution_Abs': np.abs(pca.components_[1]) if len(pca.components_) > 1 else 0
    })
    
    # Calculate combined importance score
    importance_df['Combined_Score'] = (
        0.4 * importance_df['PCA_Importance'] + 
        0.4 * importance_df['Price_Correlation_Abs'] + 
        0.2 * importance_df['PC1_Contribution_Abs']
    )
    
    # Sort by combined score
    importance_df = importance_df.sort_values('Combined_Score', ascending=False)
    
    # Create visualization
    fig, axes = plt.subplots(2, 1, figsize=(15, 12))
    
    # Combined importance ranking
    bars = axes[0].barh(importance_df['Feature'], importance_df['Combined_Score'])
    axes[0].set_title('Feature Importance Ranking (Combined Score)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Importance Score')
    axes[0].grid(True, alpha=0.3)
    
    # Add score values as text
    for i, v in enumerate(importance_df['Combined_Score']):
        axes[0].text(v + 0.01, i, f'{v:.3f}', va='center', ha='left', fontweight='bold')
    
    # Color bars by importance level
    colors = plt.cm.RdYlGn(importance_df['Combined_Score'] / importance_df['Combined_Score'].max())
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Multi-metric comparison
    metrics = ['PCA_Importance', 'Price_Correlation_Abs', 'PC1_Contribution_Abs']
    x = np.arange(len(importance_df))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        values = importance_df[metric].values
        axes[1].bar(x + i*width, values, width, label=metric.replace('_', ' '), alpha=0.8)
    
    axes[1].set_xlabel('Features')
    axes[1].set_ylabel('Normalized Importance')
    axes[1].set_title('Multi-Metric Feature Importance Comparison', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(importance_df['Feature'], rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('feature_importance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return importance_df

def print_analysis_summary(importance_df, pca, cumulative_variance):
    """Print a comprehensive analysis summary"""
    print("\n" + "="*80)
    print("📊 TECHNICAL INDICATORS ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"\n🔍 PCA ANALYSIS:")
    print(f"   • Total features analyzed: {len(importance_df)}")
    print(f"   • PC1 explains {pca.explained_variance_ratio_[0]:.1%} of variance")
    print(f"   • PC2 explains {pca.explained_variance_ratio_[1]:.1%} of variance")
    print(f"   • First 3 components explain {cumulative_variance[2]:.1%} of variance")
    
    # Find components needed for different variance thresholds
    components_80 = np.argmax(cumulative_variance >= 0.80) + 1
    components_90 = np.argmax(cumulative_variance >= 0.90) + 1
    components_95 = np.argmax(cumulative_variance >= 0.95) + 1
    
    print(f"   • Components needed for 80% variance: {components_80}")
    print(f"   • Components needed for 90% variance: {components_90}")
    print(f"   • Components needed for 95% variance: {components_95}")
    
    print(f"\n🏆 TOP 5 MOST IMPORTANT FEATURES:")
    for i, row in importance_df.head(5).iterrows():
        print(f"   {i+1}. {row['Feature']:<15} (Score: {row['Combined_Score']:.3f})")
    
    print(f"\n📉 BOTTOM 3 FEATURES (Consider for removal):")
    for i, row in importance_df.tail(3).iterrows():
        print(f"   • {row['Feature']:<15} (Score: {row['Combined_Score']:.3f})")
    
    print(f"\n💡 RECOMMENDATIONS:")
    
    # High importance features
    high_importance = importance_df[importance_df['Combined_Score'] > 0.7]['Feature'].tolist()
    medium_importance = importance_df[
        (importance_df['Combined_Score'] > 0.4) & 
        (importance_df['Combined_Score'] <= 0.7)
    ]['Feature'].tolist()
    low_importance = importance_df[importance_df['Combined_Score'] <= 0.4]['Feature'].tolist()
    
    print(f"   🔥 ESSENTIAL features ({len(high_importance)}): {', '.join(high_importance)}")
    print(f"   ⚡ IMPORTANT features ({len(medium_importance)}): {', '.join(medium_importance)}")
    if low_importance:
        print(f"   ⚠️  CONSIDER REMOVING ({len(low_importance)}): {', '.join(low_importance)}")
    
    print(f"\n📈 CORRELATION INSIGHTS:")
    top_corr = importance_df.nlargest(3, 'Price_Correlation_Abs')
    print(f"   • Highest price correlation: {top_corr.iloc[0]['Feature']} ({top_corr.iloc[0]['Price_Correlation_Abs']:.3f})")
    print(f"   • Second highest: {top_corr.iloc[1]['Feature']} ({top_corr.iloc[1]['Price_Correlation_Abs']:.3f})")
    print(f"   • Third highest: {top_corr.iloc[2]['Feature']} ({top_corr.iloc[2]['Price_Correlation_Abs']:.3f})")
    
    print(f"\n🎯 OPTIMAL FEATURE SET SUGGESTION:")
    optimal_features = importance_df.head(5)['Feature'].tolist()  # Top 5 features
    print(f"   For optimal performance/complexity balance, consider using:")
    print(f"   {optimal_features}")
    print(f"   This would explain ~{cumulative_variance[4]:.1%} of the variance with {len(optimal_features)} features.")
    
    print("\n" + "="*80)

def main():
    """Main analysis function"""
    print("🚀 Starting Technical Indicators Analysis for XAUUSD")
    print("="*60)
    
    try:
        # Load and prepare data
        df, feature_columns = load_and_prepare_data()
        print(f"✅ Data loaded successfully. Shape: {df.shape}")
        
        # Calculate correlations
        pearson_corr, spearman_corr, analysis_data = calculate_correlations(df, feature_columns)
        print("✅ Correlations calculated")
        
        # Create correlation heatmaps
        price_corr_pearson, price_corr_spearman = create_correlation_heatmaps(
            pearson_corr, spearman_corr, feature_columns)
        print("✅ Correlation heatmaps created")
        
        # Perform PCA analysis
        pca, X_scaled, cumulative_variance, scaler = perform_pca_analysis(analysis_data, feature_columns)
        print("✅ PCA analysis completed")
        
        # Create PCA visualizations
        pc1_contributions, pc2_contributions = create_pca_visualizations(
            pca, feature_columns, cumulative_variance)
        print("✅ PCA visualizations created")
        
        # Feature importance analysis
        importance_df = create_feature_importance_analysis(pca, feature_columns, price_corr_pearson)
        print("✅ Feature importance analysis completed")
        
        # Print comprehensive summary
        print_analysis_summary(importance_df, pca, cumulative_variance)
        
        # Save results to CSV
        importance_df.to_csv('feature_importance_results.csv', index=False)
        pearson_corr.to_csv('pearson_correlations.csv')
        spearman_corr.to_csv('spearman_correlations.csv')
        
        print(f"\n💾 Results saved:")
        print(f"   • feature_importance_results.csv")
        print(f"   • pearson_correlations.csv") 
        print(f"   • spearman_correlations.csv")
        print(f"   • correlation_analysis.png")
        print(f"   • pca_analysis.png")
        print(f"   • feature_importance_analysis.png")
        
        return importance_df, pca, analysis_data
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    # Set style for better plots
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Run analysis
    importance_df, pca, analysis_data = main()
    
    if importance_df is not None:
        print(f"\n🎉 Analysis completed successfully!")
        print(f"📊 Check the generated plots and CSV files for detailed results.")