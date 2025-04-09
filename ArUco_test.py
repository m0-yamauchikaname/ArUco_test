import cv2
import numpy as np
from cv2 import aruco  # この行が正しいことを確認

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
    marker_num = 4 if id_mark // 100 <= 4 else 5 if id_mark // 100 == 5 else 1

    # マーカーの一辺の長さを計算
    marker_size_cm = marker_size_max / marker_num  # 面積を分割
    marker_size_cm = marker_size_cm ** 0.5  # 面積から一辺の長さを計算

    # 余白のサイズを計算
    #rect_padding_cm = marker_size_cm * marker_size_ratio
    #marker_size_cm -= rect_padding_cm  # Adjust marker size for padding

    return marker_size_cm


# カメラキャリブレーションパラメータ（例: 焦点距離、光学中心、歪み係数）
# キャリブレーションツールで取得した値を設定してください
camera_matrix = np.array([[1000, 0, 640],
                          [0, 1000, 360],
                          [0, 0, 1]], dtype=np.float32)  # fx, fy, cx, cy
dist_coeffs = np.array([0.1, -0.25, 0.001, 0.001, 0.1], dtype=np.float32)  # k1, k2, p1, p2, k3

# ArUcoマーカーの辞書を選択
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)

# カメラの初期化
cap = cv2.VideoCapture(0)  # カメラデバイスID（0は通常デフォルトカメラ）

# 歪み補正の有効/無効を切り替えるフラグ
undistort_enabled = True

print("Press 'd' to toggle distortion correction.")
print("Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("カメラから映像を取得できませんでした。")
        break

    # 歪み補正の有効/無効に応じて処理を切り替え
    if undistort_enabled:
        h, w = frame.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
        frame_processed = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
    else:
        frame_processed = frame

    # グレースケール変換
    gray = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2GRAY)

    # ArUcoマーカーの検出
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict)
    
    if ids is not None:
        # マーカーを描画
        aruco.drawDetectedMarkers(frame_processed, corners, ids)
        for i in range(len(ids)):
            # マーカーのサイズを計算
            marker_length = calculate_marker_properties(ids[i][0]) / 100  # サイズをメートル単位に変換

            # 各マーカーの座標を計算
            rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(corners[i], marker_length, camera_matrix, dist_coeffs)

            # マーカーの座標軸を描画
            cv2.drawFrameAxes(frame_processed, camera_matrix, dist_coeffs, rvecs[0], tvecs[0], marker_length / 2)

            # マーカーの座標(x, y, z)を表示
            x, y, z = tvecs[0][0]*1000  # mm単位に変換
            # マーカーのIDと座標を表示,文字の枠の色を白、文字の色を赤に変更
            cv2.putText(frame_processed, f"ID: {ids[i][0]} Pos: ({x:.2f}, {y:.2f}, {z:.2f})",
                        (int(corners[i][0][0][0]), int(corners[i][0][0][1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            
            """
            # マーカーのIDとカメラ中心からの距離を計算
            tvec = tvecs[i][0]
            distance = np.sqrt(tvec[0]**2 + tvec[1]**2 + tvec[2]**2)  # カメラ中心からの距離

            # マーカーのIDと距離を表示
            cv2.putText(frame_processed, f"ID: {ids[i][0]} Dist: {distance:.2f}m",
                        (int(corners[i][0][0][0]), int(corners[i][0][0][1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            """


    # 映像を表示
    cv2.imshow("ArUco Marker Detection", frame_processed)

    # キーボード入力を処理
    key = cv2.waitKey(1) & 0xFF
    if key == ord('d'):  # 'd'キーで歪み補正の有効/無効を切り替え
        undistort_enabled = not undistort_enabled
        print(f"Distortion correction {'enabled' if undistort_enabled else 'disabled'}.")
    elif key == ord('q'):  # 'q'キーで終了
        break

# リソース解放
cap.release()
cv2.destroyAllWindows()