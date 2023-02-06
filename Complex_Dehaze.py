from skimage.metrics.simple_metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torch
import torch.utils as utils
from torch.autograd import Variable
from ultralytics import YOLO

from DCP import *
from Encoder_Decoder_Net import Encoder, Decoder

encoder = Encoder().cuda()
decoder = Decoder().cuda()

encoder, decoder = torch.load('best.pkl')


# def psnr(img1, img2):
#     mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
#     if mse < 1.0e-10:
#         return 100
#     PIXEL_MAX = 1
#     return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


def wb(img, percent=0.5):
    img *= 255
    img.astype(np.uint8)
    r = img[:, :, 2]
    b = img[:, :, 0]
    g = img[:, :, 1]

    # 计算BGR均值
    averB = np.mean(b)
    averG = np.mean(g)
    averR = np.mean(r)

    # 计算灰度均值
    grayValue = (averR + averB + averG) / 3

    # 计算增益
    kb = grayValue / averB
    kg = grayValue / averG
    kr = grayValue / averR

    r[:] = (r[:] * kr).clip(0, 255)
    g[:] = (g[:] * kg).clip(0, 255)
    b[:] = (b[:] * kb).clip(0, 255)
    img = img.astype(np.float32)
    return img / 255


def get_entropy(img):
    img_ = np.copy(img)
    img_ *= 255
    img_ = cv2.cvtColor(img_.astype(np.float32), cv2.COLOR_BGR2GRAY).astype(np.uint8)
    x, y = img_.shape[0:2]
    img_ = cv2.resize(img_, (100, 100))  # 缩小的目的是加快计算速度
    tmp = []
    for i in range(256):
        tmp.append(0)
    val = 0
    k = 0
    res = 0
    for i in range(len(img_)):
        for j in range(len(img_[i])):
            val = int(img_[i][j])
            tmp[val] = float(tmp[val] + 1)
            k = float(k + 1)
    for i in range(len(tmp)):
        tmp[i] = float(tmp[i] / k)
    for i in range(len(tmp)):
        if (tmp[i] == 0):
            res = res
        else:
            res = float(res - tmp[i] * (np.log(tmp[i]) / np.log(2.0)))
    return res


def readimg(path: "str"):
    IMG_SIZE = 256
    img = cv2.imread(path)
    w, h = img.shape[:2]
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    data = []
    data.append(np.array(img))
    X_hazy = torch.Tensor(data) / 255
    X_hazy_T = np.transpose(X_hazy, (0, 3, 1, 2))
    X_hazy_flat = X_hazy_T.reshape(-1, 1, IMG_SIZE, IMG_SIZE)
    return X_hazy_flat


def replaceZeroes(data):
    min_nonzero = min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data


def gamma_trans(img: "np.float32"):  # gamma函数处理
    img *= 255
    img[img > 250] = 250
    img = img.astype(np.uint8)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = np.mean(img_gray)
    gamma = np.log10(0.5) / np.log10(mean / 255)
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]  # 建立映射表
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)  # 颜色值为整数
    res = cv2.LUT(img, gamma_table).astype(np.float64) / 255  # 图片颜色查表。另外可以根据光强（颜色）均匀化原则设计自适应算法。
    return res.astype(np.float32)


def clahe(image):
    image *= cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX) * 255
    image = image.astype(np.uint8)
    b, g, r = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)
    image_clahe = cv2.merge([b, g, r])
    return image_clahe.astype(np.float64) / 255


def net_run(img):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        return
    IMG_SIZE = 256
    H, W = img.shape[:2]
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    data = np.array([np.array(img)])
    X_hazy = torch.Tensor(data) / 255
    X_hazy_T = np.transpose(X_hazy, (0, 3, 1, 2))
    X_hazy_flat = X_hazy_T.reshape(-1, 1, IMG_SIZE, IMG_SIZE)

    hazy_loader = torch.utils.data.DataLoader(dataset=X_hazy_flat, batch_size=1, shuffle=False)
    dehazed_output = []
    for hazy in hazy_loader:
        hazy_image = Variable(hazy).cuda()
        encoder_op = encoder(hazy_image)
        output = decoder(encoder_op)
        output = output.cpu()
        output = output.detach()
        dehazed_output.append(output)
    X_dehazed = dehazed_output
    X_dehazed = torch.stack(X_dehazed)
    X_dehazed = X_dehazed.view(-1, 1, 256, 256)
    X_dehazed = X_dehazed.view(-1, 3, 256, 256)
    X_dehazed = X_dehazed.permute(0, 2, 3, 1)
    Dehazed_img = cv2.resize(X_dehazed[0].cpu().data.numpy().astype(np.float32), (W, H))

    return Dehazed_img.astype(np.float32)


def complex_dehaze(src, ratio=None):
    # ratio:"Net weight"
    D = net_run(src)
    img = src.astype('float64') / 255
    gray = cv2.cvtColor(img.astype('float32'), cv2.COLOR_BGR2GRAY).astype('float64') / 255
    atmo = get_atmo(cv2.GaussianBlur(D.astype(np.float32), (5, 5), 0, 0))
    trans = get_trans(img, atmo)
    t_guided = cv2.max(guided_filter(trans, gray, 20, 0.0001), 0.25)
    J = recover(img, t_guided, atmo).astype(np.float32)
    J = (wb(J) + wb(DCP_dehaze(src))) / 2
    entropy = [get_entropy(J), get_entropy(D)]
    if ratio is None:
        ratio = entropy[1] / (entropy[0] + entropy[1])
    # print(f"Ratio = {ratio}")
    result = J * (1 - ratio) + D * ratio
    result = cv2.normalize(result, None, 0, 1, cv2.NORM_MINMAX)
    result = (gamma_trans(result))

    return result


if __name__ == "__main__":
    path = r"G:\Digital Image Processing\Dense_Haze_NTIRE19\hazy\20_hazy.png"
    src = cv2.imread(path)
    H, W = src.shape[:2]
    # DE_SIZE = 256
    # if H > DE_SIZE and W > DE_SIZE:
    #     if H >= W:
    #         W = W * (DE_SIZE / W)
    #         H = H * (DE_SIZE / W)
    #     else:
    #         W = W * (DE_SIZE / H)
    #         H = H * (DE_SIZE / H)
    #     H, W = int(H), int(W)
    # src = cv2.resize(src, (W, H))
    result = complex_dehaze(src)
    result = (cv2.cvtColor(result.astype(np.float32), cv2.COLOR_BGR2RGB)*255).astype(np.uint8)


    # plt.subplot(121)
    # plt.title("Original")
    # plt.imshow(cv2.cvtColor(src.astype(np.uint8), cv2.COLOR_BGR2RGB))
    # plt.subplot(122)
    # plt.title("DCP with Net")
    # plt.imshow(result)

    plt.subplot(131)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(src.astype(np.uint8), cv2.COLOR_BGR2RGB))
    plt.subplot(133)
    plt.title("DCP with Net")
    plt.imshow(result)
    plt.subplot(132)
    plt.title("DCP")
    dcpim = cv2.normalize(gamma_trans(DCP_dehaze(src)).astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
    plt.imshow(cv2.cvtColor(dcpim.astype(np.float32), cv2.COLOR_BGR2RGB))
    plt.show()


    def getpsnr():
        import os
        path = "G:/Digital Image Processing/Test_img/"
        gtpath = "G:/Digital Image Processing/Dense_Haze_NTIRE19/GT"
        file_list = os.listdir(path)
        for i, fi in enumerate(file_list):
            old_dir = os.path.join(path, fi)
            # print(old_dir)
            nf = str(fi.split(".")[0]) + "_GT.png"
            new_dir = os.path.join(gtpath, nf)
            # print(new_dir)
            src = cv2.imread(old_dir)
            gt = cv2.imread(new_dir)

            imd = (cv2.normalize(gamma_trans(DCP_dehaze(src)).astype(np.float32), None, 0, 1,
                                 cv2.NORM_MINMAX) * 255).astype(np.uint8)
            imc = (complex_dehaze(src) * 255).astype(np.uint8)

            psnrdcp = psnr(gt, imd)
            psnrcd = psnr(gt, imc)
            ssimdcp = ssim(gt, imd, multichannel=True)
            ssimcd = ssim(gt, imc, multichannel=True)
            print(psnrdcp, psnrcd, ssimdcp, ssimcd)

    # getpsnr()
