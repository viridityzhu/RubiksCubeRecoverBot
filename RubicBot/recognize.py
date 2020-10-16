from PIL import Image
from src.models import get_unet_96
import albumentations as A
import os
from src.utils import *
# 上黄右橘前绿下白左红后蓝
# 1：上黄左红前绿 2：上白左蓝右橘色
# flag=0 第一次识别上左前   =1 第二次识别下后右


def turn(destroyWindowList, createWindowList):
    place = [(100, 500), (525, 500), (950, 500)]
    i = 0
    for window_name in destroyWindowList:
        cv2.destroyWindow(window_name)
    for window_name in createWindowList:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(window_name, place[i][0], place[i][1])
        i += 1


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


def rec():
    SUCCESS_FLAG = [0, 0]
    FOLDERPATH = os.getcwd()
    OUTPUT = FOLDERPATH + '/output/'
    SIZE = 128
    MODEL_PATH = FOLDERPATH + '/models/seg-unet96-rds4-s128-e2-2-86.h5'
    TURN = 1

    model = get_unet_96(input_shape=(SIZE, SIZE, 1), num_classes=2)
    model.load_weights(MODEL_PATH)

    video = cv2.VideoCapture(0)
    capturingWin = "Capturing"
    cv2.namedWindow(capturingWin, cv2.WINDOW_NORMAL)
    cv2.moveWindow(capturingWin, 200, 100)
    maskWin = "Mask"
    cv2.namedWindow(maskWin, cv2.WINDOW_NORMAL)
    cv2.moveWindow(maskWin, 800, 100)
    turnOne = ["Left_red", "Top_yellow", "Right_green"]
    turnTwo = ["Left_blue", "Top_white", "Right_orange"]
    turn([], turnOne)

    while True:
        _, frame = video.read()
        frame = A.CenterCrop(height=480, width=480)(image=frame)['image']
        copy_frame = frame.copy()

        gray = Image.fromarray(frame, 'RGB').convert('L')

        im = gray.resize((SIZE, SIZE))
        img_array = np.array(im) / 255
        img_array = np.reshape(img_array, (SIZE, SIZE, 1))

        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)

        mask = prediction[0][:, :, 0]
        mask = cv2.resize(mask, (480, 480))

        center = prediction[0][:, :, 1]
        center = cv2.resize(center, (480, 480))

        copy_frame[mask < 0.5, :] = 0
        copy_frame[center > 0.5] = 0

        mask[mask < 0.5] = 0
        mask[mask != 0] = 255
        mask = np.array(mask, np.uint8)

        center = np.array(center * 255, np.uint8)

        mask_copy = mask.copy()
        binary_mask = mask
        binary_center = cv2.threshold(
            center, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        approx = get_corners(binary_mask, alpha=0.03)

        cv2.imshow(capturingWin, frame)

        key = cv2.waitKey(1)
        if key == ord('n'):
            TURN = 2
            turn(turnOne, turnTwo)
        elif key == ord('b'):
            TURN = 1
            turn(turnTwo, turnOne)
        elif key == ord('q'):
            break

        if approx is None:
            # print('Bad frame')
            continue

        center_point = get_center_point(binary_center)

        if center_point is None:
            # print('Bad center point')
            continue

        points = [(p[0][0], p[0][1]) for p in approx]

        rect_left = np.array([points[1], center_point, points[3], points[2]])
        rect_top = np.array([points[0], points[5], center_point, points[1]])
        rect_forward = np.array(
            [center_point, points[5], points[4], points[3]])

        side_left = four_points_resize(frame, rect_left, 256)
        side_top = four_points_resize(frame, rect_top, 256)
        side_forward = four_points_resize(frame, rect_forward, 256)

        cv2.circle(mask, points[5], 5, (0, 255, 0), -1)

        cv2.drawContours(mask, [approx], 0, (147, 0, 255), 3)
        cv2.circle(mask, center_point, 5, (147, 0, 255), -1)

        cv2.drawContours(frame, [rect_left], 0, (147, 0, 255), 3)
        cv2.drawContours(frame, [rect_top], 0, (255, 0, 0), 3)
        cv2.drawContours(frame, [rect_forward], 0, (0, 255, 0), 3)

        cv2.line(mask, points[1], points[0],
                 (0, 255, 0), thickness=3, lineType=8)
        cv2.line(mask, center_point, points[5],
                 (0, 255, 0), thickness=3, lineType=8)

        if is_hexagon_valid(points, center_point, epsilon=10):
            mask[mask == 0] = 45
            if TURN == 1:
                cv2.imshow(turnOne[0], side_left)
                cv2.imshow(turnOne[1], side_top)
                cv2.imshow(turnOne[2], side_forward)
                cv2.imwrite(OUTPUT + '/' + turnOne[0] + '.jpg', side_left)
                cv2.imwrite(OUTPUT + '/' +
                            turnOne[1] + '.jpg', rotate_bound(side_top, 90))
                cv2.imwrite(OUTPUT + '/' + turnOne[2] + '.jpg', side_forward)
                SUCCESS_FLAG[0] = 1
            elif TURN == 2:
                cv2.imshow(turnTwo[0], side_left)
                cv2.imshow(turnTwo[1], side_top)
                cv2.imshow(turnTwo[2], side_forward)
                cv2.imwrite(OUTPUT + '/' +
                            turnTwo[0] + '.jpg', rotate_bound(side_left, 180))
                cv2.imwrite(OUTPUT + '/' + turnTwo[1] + '.jpg', side_top)
                cv2.imwrite(
                    OUTPUT + '/' + turnTwo[2] + '.jpg', rotate_bound(side_forward, 180))
                SUCCESS_FLAG[1] = 1

            cv2.imshow(capturingWin, frame)
            key = cv2.waitKey(1)

        cv2.imshow(maskWin, binary_mask)
        key = cv2.waitKey(1)

    video.release()
    cv2.destroyAllWindows()
    if SUCCESS_FLAG[0] == 1 and SUCCESS_FLAG[1] == 1:
        return True
    else:
        return False

if __name__ == '__main__':
    rec()
