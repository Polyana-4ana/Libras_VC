import cv2
import mediapipe as mp
import joblib
import numpy as np

# Carrega modelo treinado
modelo = joblib.load(
    "models/libras_model.pkl"
)

# MediaPipe
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=1
)

mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

while True:

    sucesso, img = cap.read()

    if not sucesso:
        break

    rgb = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    resultado = hands.process(rgb)

    palavra = ""

    if resultado.multi_hand_landmarks:

        for handLms in resultado.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                img,
                handLms,
                mp_hands.HAND_CONNECTIONS
            )

            dados=[]

            for lm in handLms.landmark:

                dados.extend(
                    [lm.x,lm.y,lm.z]
                )

            dados=np.array(dados).reshape(1,-1)

            palavra = modelo.predict(
                dados
            )[0]

    cv2.putText(
        img,
        f"Previsao: {palavra}",
        (10,50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.imshow(
        "IA Libras",
        img
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()

cv2.destroyAllWindows()