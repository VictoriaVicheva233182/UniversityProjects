# ─── Imports ─────────────────────────────────────────────────────────

import tensorflow as tf
from tensorflow.keras import backend as K

# ─── Model functions ─────────────────────────────────────────────────

def f1(y_true, y_pred):
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = TP / (Positives+K.epsilon())
        return recall
    
    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = TP / (Pred_Positives+K.epsilon())
        return precision
    
    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)
    
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def load_model(model_path: str) -> tf.keras.Model:
    """Load a U‑Net/ResNet model with custom F1 metric."""
    return tf.keras.models.load_model(
        model_path,
        custom_objects={"f1": f1}
    )
