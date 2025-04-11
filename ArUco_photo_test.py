import cv2
import numpy as np
from cv2 import aruco  # この行が正しいことを確認
import matplotlib.pyplot as plt


print(cv2.__version__)

# 実行方法
##必要なライブラリをインストールします。
##pip install opencv-python opencv-contrib-python numpy
##上記のコードをPythonファイル（例: aruco_detection.py）として保存します。
##実行します。
##カメラ映像にArUcoマーカーを映すと、マーカーのIDと座標が表示されます。


def calculate_marker_properties(id_mark):
    """
    マーカーのIDに基づいて、マーカーのサイズと余白を計算する関数。

    Args:
        id_mark (int): マーカーのID。

    Returns:
        tuple: マーカーの一辺の長さ (cm) と余白のサイズ (cm)。
    """
    marker_size_max = 4.0  # 最大面積 (cm²)
    #marker_size_ratio = 0.2  # 余白の比率

    # マーカーの枚数を決定
    # marker_num = 4 if id_mark // 100 <= 4 else 5 if id_mark // 100 == 5 else 1
    marker_num = 4

    # マーカーの一辺の長さを計算
    marker_size_cm = marker_size_max / marker_num  # 面積を分割
    marker_size_cm = marker_size_cm ** 0.5  # 面積から一辺の長さを計算

    # 余白のサイズを計算
    #rect_padding_cm = marker_size_cm * marker_size_ratio
    #marker_size_cm -= rect_padding_cm  # Adjust marker size for padding

    return marker_size_cm


# カメラキャリブレーションパラメータ（例: 焦点距離、光学中心、歪み係数）
camera_matrix = np.array([[1000, 0, 640],
                          [0, 1000, 360],
                          [0, 0, 1]], dtype=np.float32)  # fx, fy, cx, cy
dist_coeffs = np.array([0.1, -0.25, 0.001, 0.001, 0.1], dtype=np.float32)  # k1, k2, p1, p2, k3

# ArUcoマーカーの辞書を選択
aruco_dict = aruco.extendDictionary(75, 3)
# aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)

# カメラの初期化
cap = cv2.VideoCapture(1)  # カメラデバイスID（0は通常デフォルトカメラ）

# 歪み補正の有効/無効を切り替えるフラグ
undistort_enabled = True

print("Press 'd' to toggle distortion correction.")
print("Press 'c' to capture a photo.")
print("Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("カメラから映像を取得できませんでした。")
        break

    # 映像を表示
    cv2.imshow("Camera Feed", frame)

    # キーボード入力を処理
    key = cv2.waitKey(1) & 0xFF
    if key == ord('d'):  # 'd'キーで歪み補正の有効/無効を切り替え
        undistort_enabled = not undistort_enabled
        print(f"Distortion correction {'enabled' if undistort_enabled else 'disabled'}.")
    elif key == ord('c'):  # 'c'キーで写真を撮影して処理
        print("Photo captured. Processing...")
        if undistort_enabled:
            h, w = frame.shape[:2]
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
            frame_processed = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
        else:
            frame_processed = frame

        # 撮影した画像を保存
        cv2.imwrite("image.jpg", frame_processed)
        print("Saved image as 'image.jpg'.")

        # 保存した画像を読み込んで処理
        image = cv2.imread("image.jpg")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # ArUcoマーカーの検出
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict)
        if ids is not None:
            marker_size_cm =0.8
            marker_length = marker_size_cm / 100  # cmをmに変換
            # マーカーの位置姿勢推定
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

            # マーカーを描画
            for i in range(len(ids)):
                aruco.drawDetectedMarkers(image, corners, ids)
                # マーカーの座標軸を描画
                cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], marker_length / 2)
            # マーカーの座標(x, y, z)を表示
            # マーカーのIDと座標を表示,文字の枠の色を白、文字の色を赤に変更
            for i in range(len(ids)):
                x, y, z = tvecs[i][0] * 1000
                cv2.putText(image, f"ID: {ids[i][0]} Pos: ({x:.2f}, {y:.2f}, {z:.2f})",
                            (int(corners[i][0][0][0]), int(corners[i][0][0][1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
            # 処理結果を保存
            cv2.imwrite("edited_image.jpg", image)
            print("Saved processed image as 'edited_image.jpg'.")
        else:
            print("No markers detected.")
    elif key == ord('q'):  # 'q'キーで終了
        break

# リソース解放
cap.release()
cv2.destroyAllWindows()