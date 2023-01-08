import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import imutils
import os
import math

def detectAndDescribe(image, top, bot, left, right):

    Img = cv.copyMakeBorder(image, top, bot, left, right, cv.BORDER_CONSTANT, value=(0, 0, 0))
    imggray = cv.cvtColor(Img, cv.COLOR_BGR2GRAY)
    sift = cv.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(imggray, None)
    return kp,des

def matchKeypoints(kp1, kp2, des1, des2, ratio, reprojThresh,MIN_MATCH_COUNT):#FLANN_INDEX_KDTREE
    # index_params = dict(algorithm=1, trees=5)
    # search_params = dict(checks=50)
    # flann = cv.FlannBasedMatcher(index_params, search_params)  # FlannBasedMatcher
    # matches = flann.knnMatch(des1, des2, k=2)
    matcher = cv.BFMatcher()
    matches = matcher.knnMatch(des1, des2, 2)
    matchesMask = [[0, 0] for i in range(len(matches))]
    good = []

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < ratio * n.distance:
            good.append(m)
            matchesMask[i] = [1, 0]


    if len(good) > MIN_MATCH_COUNT:
        Part1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        Part2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv.findHomography(Part1, Part2, cv.RANSAC, reprojThresh)

        return (matches, H, mask,matchesMask)

    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None
        return matchesMask
    # top, bot, left, right = 100, 100, 0, 500
def swap(a, b):
    return b,a

def cutBlack(pic):
    rows, cols = np.where(pic[:,:,0] !=0)
    min_row, max_row = min(rows), max(rows) +1
    min_col, max_col = min(cols), max(cols) +1
    return pic[min_row:max_row,min_col:max_col,:]

def stitch(images,ratio = 0.7, reprojThresh = 5.0,top = 100, bot = 100, left = 0, right = 5,MIN_MATCH_COUNT = 15,mapping_strategy = "max"):
    (imageA,imageB) = images

    (kp1, des1) = detectAndDescribe(imageA,top, bot, left, right)
    (kp2, des2) = detectAndDescribe(imageB,top, bot, left, right)

    M = matchKeypoints(kp1,kp2, des1, des2,ratio, reprojThresh,MIN_MATCH_COUNT = MIN_MATCH_COUNT)

    if M is None:
        return None
    else:

        (matches, H, mask,matchesMask) = M

        result = cv.warpPerspective(imageB, np.array(H), ((imageA.shape[1] + imageB.shape[1]), (imageA.shape[0] + imageB.shape[0])),
                                     flags=cv.WARP_INVERSE_MAP)
        if np.size(cutBlack(result)) < np.size(imageB) * 0.8:
            print("圖片位置不對，將進行調換")
            # 调换图片
            kp1, kp2 = swap(kp1, kp2)
            imageA, imageB = swap(imageA, imageB)
            des1,des2 = swap(des1,des2)
            M = matchKeypoints(kp1, kp2, des1, des2, ratio, reprojThresh,MIN_MATCH_COUNT = MIN_MATCH_COUNT)

            if M is None:
                return None
            (matches, H, mask,matchesMask) = M
            result = cv.warpPerspective(imageB, np.array(H),((imageA.shape[1] + imageB.shape[1]), (imageA.shape[0] + imageB.shape[0])),
                                        flags=cv.WARP_INVERSE_MAP)

        drawmatch(imageA,imageB,kp1,kp2,matches,matchesMask)

        if mapping_strategy == "max":
            result[0:imageA.shape[0], 0:imageA.shape[1]] = np.maximum(imageA,
                                                                      result[0:imageA.shape[0], 0:imageA.shape[1]])
        else :
            if  mapping_strategy == "rz":
                result = replace_zero(imageA,result)
        result = cutBlack(result)
    return result

def replace_zero(imageA,result):
    cols = imageA.shape[0]
    rows = imageA.shape[1]
    x,y,_ = np.where(result[0:cols,0:rows] == 0)
    point = np.unique(np.array([x, y]).T, axis=0)
    for p in point:
        result[p[0],p[1]] = imageA[p[0],p[1]]
    return result


def drawmatch(img1,img2,kp1,kp2,matches,matchesMask):
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=0)

    img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
    img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)


    img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
    plt.imshow(img3, )
    plt.show()




def cylindrical_projection(img, f=700):
    rows = img.shape[0]
    cols = img.shape[1]

    # f = cols / (2 * math.tan(np.pi / 8))

    blank = np.zeros_like(img)
    center_x = int(cols / 2)
    center_y = int(rows / 2)

    for y in range(rows):
        for x in range(cols):
            theta = math.atan((x - center_x) / f)
            point_x = int(f * math.tan((x - center_x) / f) + center_x)
            point_y = int((y - center_y) / math.cos(theta) + center_y)

            if point_x >= cols or point_x < 0 or point_y >= rows or point_y < 0:
                pass
            else:
                blank[y, x, :] = img[point_y, point_x, :]
    return blank



def adjust_img_black(img):
    add_black = cv.copyMakeBorder(img, 20, 5, 20, 20,cv.BORDER_CONSTANT, (0, 0, 0))
    gray = cv.cvtColor(add_black, cv.COLOR_BGR2GRAY)
    thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY)[1]

    cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
                            cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv.contourArea)

    mask = np.zeros(thresh.shape, dtype="uint8")
    (x, y, w, h) = cv.boundingRect(c)
    cv.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    minRect = mask.copy()
    sub = mask.copy()

    # keep looping until there are no non-zero pixels left in the
    # subtracted image
    while cv.countNonZero(sub) > 0:
        # erode the minimum rectangular mask and then subtract
        # the thresholded image from the minimum rectangular mask
        # so we can count if there are any non-zero pixels left
        minRect = cv.erode(minRect, None)
        sub = cv.subtract(minRect, thresh)

        cnts = cv.findContours(minRect.copy(), cv.RETR_EXTERNAL,
                                cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv.contourArea)
        (x, y, w, h) = cv.boundingRect(c)

    image = add_black[y:y + h, x:x + w]
    return image

def main(path,set_plane_size,is_futer_cut_black = False,mapping_strategy = "max"):
    name_list = []
    name = locals()
    i = 0
    for filename in os.listdir(path):
        i += 1
        name["img_" + str(i)] = cv.imread(path + '\\' + filename)
        name["img_" + str(i)] = cylindrical_projection(name["img_" + str(i)], set_plane_size)  # 設定投影平面
        name_list.append("img_" + str(i))
        # img_list.append(name["img_"+str(i)])
    img0 = None
    bb = len(name_list)
    log_list = []
    while name_list != []:
        n = 0

        for imgname in name_list:
            if img0 is None:
                img0 = name[imgname]
                name_list.remove(imgname)
                continue

            res = stitch([img0, name[imgname]], MIN_MATCH_COUNT=100,mapping_strategy = mapping_strategy)
            # print(res)
            if res is None:
                n += 1
                break
            else:
                img0 = res
                img0 = np.array(res)
                name_list.remove(imgname)
        log_list.append(n)
        if sum(log_list) > 20:
            print("some img can not stitch")
            break

        if n == bb:
            print("all img can not stitch")
            break
        if n == 1 and len(name_list) == 1:
            print("only one img can not stitch")
            break

    res = cv.cvtColor(img0, cv.COLOR_BGR2RGB)
    if is_futer_cut_black:
        res = adjust_img_black(res)
        plt.figure()
        plt.imshow(res)
        plt.savefig(path)
        plt.show()
    else:
        plt.figure()
        plt.imshow(res)
        plt.savefig(path)
        plt.show()

if __name__ == '__main__':
    path = os.getcwd()
    path = path + "\\input_img\\tree"#設定 資料夾  #Hill tree scottsdale river
    main(path,900,mapping_strategy = "rz",is_futer_cut_black=False) # mapping_strategy "max" or "rz" #is_futer_cut_black ture or false

















