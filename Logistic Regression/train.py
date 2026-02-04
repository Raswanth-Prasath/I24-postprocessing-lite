import numpy as np
      import pickle
      from sklearn.linear_model import LogisticRegression
      from sklearn.preprocessing import StandardScaler
      from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
      from sklearn.metrics import roc_auc_score, average_precision_score
      from datetime import datetime

      # Load the full 47-feature dataset
      data = np.load('Logistic Regression/training_dataset_advanced.npz', allow_pickle=True)
      X = data['X']
      y = data['y']
      all_features = list(data['feature_names'])

      # Top 10 consensus features (from full 47-feature analysis)
      top10_features = [
          'y_diff',
          'time_gap',
          'projection_error_x_max',
          'length_diff',
          'width_diff',
          'projection_error_y_max',
          'bhattacharyya_coeff',
          'projection_error_x_mean',
          'curvature_diff',
          'projection_error_x_std',
      ]

      print('=' * 70)
      print('TRAINING TOP 10 CONSENSUS FEATURES MODEL')
      print('=' * 70)
      print(f'\nSelected features:')
      for i, f in enumerate(top10_features, 1):
          print(f'  {i:2}. {f}')

      # Get indices of selected features
      indices = [all_features.index(f) for f in top10_features]
      X_subset = X[:, indices]

      print(f'\nDataset: {X_subset.shape[0]} samples, {X_subset.shape[1]} features')

      # Scale features
      scaler = StandardScaler()
      X_scaled = scaler.fit_transform(X_subset)

      # Split data
      X_train, X_test, y_train, y_test = train_test_split(
          X_scaled, y, test_size=0.2, random_state=42, stratify=y
      )

      # Train model
      lr = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0)
      lr.fit(X_train, y_train)

      # Evaluate
      y_proba_train = lr.predict_proba(X_train)[:, 1]
      y_proba_test = lr.predict_proba(X_test)[:, 1]

      train_roc = roc_auc_score(y_train, y_proba_train)
      test_roc = roc_auc_score(y_test, y_proba_test)
      train_ap = average_precision_score(y_train, y_proba_train)
      test_ap = average_precision_score(y_test, y_proba_test)

      # Cross-validation
      cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
      cv_scores = cross_val_score(lr, X_scaled, y, cv=cv, scoring='roc_auc')

      print(f'\n' + '=' * 70)
      print('PERFORMANCE METRICS')
      print('=' * 70)
      print(f'Train ROC-AUC: {train_roc:.4f}')
      print(f'Test ROC-AUC:  {test_roc:.4f}')
      print(f'Train AP:      {train_ap:.4f}')
      print(f'Test AP:       {test_ap:.4f}')
      print(f'CV ROC-AUC:    {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')

      # Retrain on full data for deployment
      scaler_full = StandardScaler()
      X_scaled_full = scaler_full.fit_transform(X_subset)
      lr_full = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0)
      lr_full.fit(X_scaled_full, y)

      # Show coefficients
      print(f'\n' + '=' * 70)
      print('FEATURE COEFFICIENTS')
      print('=' * 70)
      print(f'{\"Feature\":<28} {\"Coefficient\":>12}')
      print('-' * 42)
      sorted_idx = np.argsort(np.abs(lr_full.coef_[0]))[::-1]
      for idx in sorted_idx:
          print(f'{top10_features[idx]:<28} {lr_full.coef_[0][idx]:>+12.4f}')

      # Save model artifacts
      artifacts = {
          'model': lr_full,
          'scaler': scaler_full,
          'feature_names': top10_features,
          'n_features': len(top10_features),
          'metrics': {
              'train_roc_auc': train_roc,
              'test_roc_auc': test_roc,
              'train_ap': train_ap,
              'test_ap': test_ap,
              'cv_roc_auc_mean': cv_scores.mean(),
              'cv_roc_auc_std': cv_scores.std()
          },
          'created': datetime.now().isoformat()
      }

      output_path = 'Logistic Regression/model_artifacts/consensus_top10_full47.pkl'
      with open(output_path, 'wb') as f:
          pickle.dump(artifacts, f)

      print(f'\n✓ Model saved to: {output_path}')
      ")