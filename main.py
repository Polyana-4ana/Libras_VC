import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

frase = ""
ultima_palavra = ""
tempo_ultima = time.time()

def detectar_gesto(hand_landmarks):
    dedos = []
    tips_ids = [8, 12, 16, 20]

    for tip in tips_ids:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            dedos.append(1)
        else:
            dedos.append(0)

    total = sum(dedos)

    # Mapeamento simples (você pode mudar depois)
    if total == 0:
        return "OI"
    elif total == 1:
        return "TUDO"
    elif total == 2:
        return "BEM"
    elif total == 3:
        return "VOCÊ"
    else:
        return ""

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img_rgb)

    palavra_atual = ""

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
            palavra_atual = detectar_gesto(handLms)

    # lógica de frase
    if palavra_atual != "" and palavra_atual != ultima_palavra:
        if time.time() - tempo_ultima > 1:
            frase += palavra_atual + " "
            tempo_ultima = time.time()
        ultima_palavra = palavra_atual

    cv2.putText(img, f"Palavra: {palavra_atual}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(img, f"Frase: {frase}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv2.imshow("Libras Demo", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()