from .models import create_model

classification_model = create_model()

history = classification_model.fit(
    X_train_tokens_padded, 
    y_train,
    batch_size=8,
    epochs=20,
    validation_data=(X_test_tokens_padded, y_test)
)