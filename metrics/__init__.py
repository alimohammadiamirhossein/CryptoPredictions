from .metrics import accuracy_score, f1_score ,recall_score, precision_score, classification_report, confusion_matrix, rmse


METRICS = {     'accuracy_score': accuracy_score,
                'f1_score': f1_score,
                'recall_score': recall_score,
                'precision_score': precision_score,
                'classification_report': classification_report,
                'confusion_matrix': confusion_matrix,
                'rmse': rmse
                }