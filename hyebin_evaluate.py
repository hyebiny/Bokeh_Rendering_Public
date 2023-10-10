import torch
import torch.nn.functional as F
import lpips
from PerceptualSimilarity.util import util


from tqdm import tqdm

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

batch_size = 1
feed_width = 1536
feed_height = 1024

def evaluate():
    tot_lpips_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0

    device = torch.device("cuda")
    loss_fn = lpips.LPIPS(net='alex').to(device)
    start = 4400
    totlength = 4693
    csv_file = "../data_train/list/valid.csv" # test.csv
    root_dir = "../ml-cvnets"
    # dataa = pd.read_csv(csv_file)
    for i in tqdm(range(start, totlength)):# range(294)):
        idx = i
        # I0 = util.load_image(root_dir + dataa.iloc[idx, 0][:])
        I0 = util.load_image(f"../data_train/bokeh/{i}.jpg")
        I1 = util.load_image(f"mobilevitv2/width_2_0_0/run_2/resize/{i}.jpg")
        img0 = util.im2tensor(I0) # RGB image from [-1,1]
        img1 = util.im2tensor(I1)
        img0 = img0.to(device)
        img1 = img1.to(device)

        h0, w0 = img0.shape[2], img0.shape[3]
        h1, w1 = img1.shape[2], img1.shape[3]
        if h0 != h1 or w0 != w1:
            # resize the input image, so that we do not get dimension mismatch errors in the forward pass
            img1 = F.interpolate(img1, (h0, w0), mode='bilinear')

        lpips_loss = loss_fn.forward(img0, img1)
        tot_lpips_loss += lpips_loss.item()

        test = util.tensor2im(img1)
        util.save_image(test, f"mobilevitv2/width_2_0_0/run_2/{i}.jpg")
        I1 = util.load_image(f"mobilevitv2/width_2_0_0/run_2/{i}.jpg")

        total_psnr += compare_psnr(I0, I1)
        total_ssim += compare_ssim(I0, I1, multichannel=True)


    print("TOTAL LPIPS:",":", tot_lpips_loss / (totlength-start))
    print("TOTAL PSNR",":", total_psnr / (totlength-start))
    print("TOTAL SSIM",":", total_ssim / (totlength-start))


if __name__ == "__main__":
    evaluate()


