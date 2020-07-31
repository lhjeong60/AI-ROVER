import cv2
import numpy as np
import os
import sys


# current_path = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(current_path)

# #%% 이미지 로딩
# img = cv2.imread("C:/Temp/a.png")
# #img = cv2.imread("C:/Temp/slope_test.jpg")
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# plt.imshow(img_gray, cmap="gray")
# plt.show()
#
# #%% Gaussian blur (노이즈 제거)
# blur_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
# plt.imshow(blur_gray, cmap="gray")
# plt.show()
#
# #%% threshold (이진화)
# th, img_th = cv2.threshold(blur_gray, 127, 255, cv2.THRESH_BINARY)
# plt.imshow(img_th, cmap="gray")
# plt.show()
#
# #%% close 모폴로지
# kernel = np.ones((5, 5), np.uint8)
# img_morph = cv2.morphologyEx(img_th.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
# plt.imshow(img_morph, cmap="gray")
# plt.show()
#
# #%% Canny detection
# edges = cv2.Canny(img_morph, 50, 200)
# plt.imshow(edges, cmap="gray")
# plt.show()
#
# #%% ROI
# mask = np.zeros_like(img_gray)
# ignore_mask_color = 255
# imshape = img_gray.shape
#
# vertices = np.array([[(130, imshape[0]),
#                       (320, 250),
#                       (420, 250),
#                       (imshape[1]-20, imshape[0])]], dtype=np.int32)
# cv2.fillPoly(mask, vertices, ignore_mask_color)
#
# plt.imshow(mask, cmap="gray")
# plt.show()
#
# #%% 관심영역만 표시
# masked_img = cv2.bitwise_and(edges, mask)
# plt.imshow(masked_img, cmap="gray")
# plt.show()
#
# #%% bird's eye view
# position1 = np.float32([[300, 250], [0, imshape[0]], [420, 250], [imshape[1], imshape[0]]])
# position2 = np.float32([[10, 10], [10, 1000], [1000, 10], [1000, 1000]])
# M = cv2.getPerspectiveTransform(position1, position2)
# dst = cv2.warpPerspective(img, M, (1100, 1100))
#
# plt.imshow(dst, cmap="gray")
# plt.show()

# %%
def get_fitline(img, f_lines):  # 대표선 구하기
    if f_lines.shape[1] == 1:
        lines = f_lines[0, :, :]
    else:
        lines = np.squeeze(f_lines)
    lines = lines.reshape(lines.shape[0] * 2, 2)
    rows, cols = img.shape[:2]
    output = cv2.fitLine(lines, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x, y = output[0], output[1], output[2], output[3]
    x1, y1 = int(((rows - 1) - y) / vy * vx + x), rows - 1
    x2, y2 = int(((rows / 2 + 100) - y) / vy * vx + x), int(rows / 2 + 100)

    result = [x1, y1, x2, y2]
    return result


# %%
def draw_fit_line(img, lines, color=[255, 0, 0], thickness=10):  # 대표선 그리기
    cv2.line(img, (lines[0], lines[1]), (lines[2], lines[3]), color, thickness)


# %% 검출된 선에 해당하는 픽셀을 색칠
def fillPoly_fit_line(edges, lines, color=[255, 0, 0], margin=100):
    x1, y1, x2, y2 = lines[0], lines[1], lines[2], lines[3]
    # 기울기 계산
    slope = (y2 - y1) / (x2 - x1)
    # ploty = np.linspace(0, edges.shape[0], edges.shape[0])

    nonzero = edges.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])

    left_bound_indices = (
            (nonzero_x > ((nonzero_y - y2) / slope + x2 - margin))
            & (nonzero_x < ((nonzero_y - y2) / slope + x2))
    )
    right_bound_indices = (
            (nonzero_x > ((nonzero_y - y2) / slope + x2))
            & (nonzero_x < ((nonzero_y - y2) / slope + x2 + margin))
    )

    img_fit = np.dstack((edges, edges, edges)) * 255

    pts_left = np.array(list(zip()))

    return img_fit


# %%
def weighted_img(img, initial_img, a=1., b=1., c=0.):  # 두 이미지 operlap 하기
    return cv2.addWeighted(initial_img, a, img, b, c)


# %%
def line_limited_angle(line_arr, img):
    # 기울기 구하기
    slope_degree = (np.arctan2(line_arr[:, 1] - line_arr[:, 3], line_arr[:, 0] - line_arr[:, 2]) * 180) / np.pi

    # 수평 기울기 제한
    line_arr = line_arr[np.abs(slope_degree) < 160]
    slope_degree = slope_degree[np.abs(slope_degree) < 160]

    # 수직 기울기 제한
    line_arr = line_arr[np.abs(slope_degree) > 95]
    slope_degree = slope_degree[np.abs(slope_degree) > 95]

    # 필터링된 직선 버리기
    L_lines, R_lines = line_arr[(slope_degree > 0), :], line_arr[(slope_degree < 0), :]
    temp = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    L_lines, R_lines = L_lines[:, None], R_lines[:, None]

    # 왼쪽, 오른쪽 각각 선이 있을때만 대표선 구하고 그리기
    if L_lines.shape[0] != 0:
        left_fit_line = get_fitline(img, L_lines)
        draw_fit_line(temp, left_fit_line)
        # fillPoly_fit_line

    if R_lines.shape[0] != 0:
        right_fit_line = get_fitline(img, R_lines)
        draw_fit_line(temp, right_fit_line)

    result = weighted_img(temp, img, a=0.8, b=5.)  # 원본 이미지에 검출된 선 overlap
    return result

def line_detect(frame):
    # =======================편한 사이즈로 재조정=========================
    # frame = cv2.resize(frame, dsize=(320, 480))

    # =======================흑백 컬러로 변환============================
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # =========================가우시안 블러 처리, 노이즈 제거=========================
    blur_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # =========================threshold 처리=====================================
    th, img_th = cv2.threshold(blur_gray, 190, 255, cv2.THRESH_BINARY)

    # =========================close morphology : 구멍이 채워지고 좀 더 뚜렷해짐=========================
    kernel = np.ones((5, 5), np.uint8)
    img_morph = cv2.morphologyEx(img_th.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    # =========================Canny 윤곽선 검출=========================
    edges = cv2.Canny(img_morph, 50, 200)

    # =========================ROI=============================================================
    mask = np.zeros_like(img_gray)
    ignore_mask_color = 255
    height, width = img_gray.shape
    vertices = np.array([[(0, height), (0, height / 3), (width, height / 3),
                          (width, height)]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_img = cv2.bitwise_and(edges, mask)

    # =========================Hough Transform을 이용한 직선 검출, 리턴된 lines는 (n, 1, 4)의 shape을 가진다.(n : 검출된 직선의 개수) =========================
    # threshold : 높을 수록 정확도는 올라가고, 적은 선을 찾음, 낮으면 많은 직선을 찾지만 대부분의 직선을 찾음
    # minLineLength : 찾을 직선의 최소 길이, maxLineGap : 선과의 최대 간격
    lines = cv2.HoughLinesP(masked_img, 1, np.pi / 180, 30, minLineLength=10, maxLineGap=20)

    if lines is None:
        return None

    if lines.shape[0] == 1:
        line_arr = lines[0, :, :]
    else:
        line_arr = lines.squeeze()

    # 기울기 구하기
    slope_degree = (np.arctan2(line_arr[:, 1] - line_arr[:, 3], line_arr[:, 0] - line_arr[:, 2]) * 180) / np.pi

    # 수평 기울기 제한
    line_arr = line_arr[np.abs(slope_degree) < 160]
    slope_degree = slope_degree[np.abs(slope_degree) < 160]

    # 수직 기울기 제한
    line_arr = line_arr[np.abs(slope_degree) > 95]
    slope_degree = slope_degree[np.abs(slope_degree) > 95]

    # 필터링된 직선 버리기
    # L_lines, R_lines = line_arr[(slope_degree > 0), :], line_arr[(slope_degree < 0), :]

    return line_arr  # L_lines, R_lines  # lines


# %% test
if __name__ == '__main__':
    videoCapture = cv2.VideoCapture("C:/MyWorkspace/final/project_4_advanced_lane_finding/test_1.avi")
    # videoCapture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    # videoCapture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    while videoCapture.isOpened():
        retval, frame = videoCapture.read()
        if not retval:
            break
        height, width = frame.shape
        lines = line_detect(frame)

        # =========================선을 찾지 못했다면, 다음 프레임으로 continue=========================
        if lines is None:
            cv2.imshow("video", frame)
            continue

        # =========================선을 하나만 찾는경우, squeeze()에 의해 선의 개수 축(0축)까지 벗겨질 수 있기 때문에 1개의 선만 찾았을 때 분할 처리
        if lines.shape[1] == 1:
            line_arr = lines[0, :, :]
        else:
            line_arr = lines.squeeze()


        # result = line_limited_angle(line_arr, frame)

        cv2.imshow("video", frame)

        key = cv2.waitKey(1)
        if key == 27:
            break

    videoCapture.release()
    cv2.destroyAllWindows()

