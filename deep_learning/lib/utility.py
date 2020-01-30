from deep_learning.lib.include import *

from typing import Optional

from deep_learning.lib.include import *


from sklearn import metrics
warnings.filterwarnings('ignore')
def f1_score(y_pred,y_true):
    y_pred = y_pred.to('cpu').detach().numpy()
    y_true = y_true.to('cpu').detach().numpy()
    y_pred[y_pred>0.5] = 1
    y_pred[y_pred<=0.5] = 0
    y_pred = y_pred.astype('int')
    y_true = y_true.astype('int')
    return metrics.f1_score(y_true,y_pred,average="samples")

def accuracy_score(y_pred,y_true):
    y_pred = y_pred.to('cpu').detach().numpy()
    y_true = y_true.to('cpu').detach().numpy()
    y_pred[y_pred>0.5] = 1
    y_pred[y_pred<=0.5] = 0
    y_pred = y_pred.astype('int')
    y_true = y_true.astype('int')
    return metrics.accuracy_score(y_true,y_pred)

def inverse_tta(a,t):
    if isinstance(t, albu.HorizontalFlip):
        a = torch.flip(a,[3])
    elif isinstance(t, albu.VerticalFlip):
        a = torch.flip(a,[2])
    elif isinstance(t, albu.Transpose):
        a = torch.transpose(a, 2, 3)
    return a

def outline_mask(df,x,y,z,image_path,save_path=None,class_names=None,plot=True,shape=None):
    """
        To show and save mask for different classes
        df = pandas dataframe
        x=column containing image name
        y=column containing run-length-encoded mask
        z=column containing class label
        imagePath=path of image
        savePath=path where to save image
        class_names=folder with class names where images with corresponding classes will be stored
    """
    if class_names is not None:
        for i in class_names:
            path = save_path + str(i)
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)
    palet = np.dot(sns.color_palette("muted",n_colors=len(class_names)),255)
    for fname in df[x].unique():
        for idx,row in enumerate(df[df[x]==fname].iterrows()):
            label = row[1][y]
            cl = row[1][z]
            img = cv2.imread(image_path+fname)
            if shape != None:
                img = cv2.resize(img,dsize=(shape[1],shape[0]),interpolation=cv2.INTER_LINEAR)
            mask_shape = img.shape
            if label is not np.nan and label != '':
                mask_label = np.zeros(mask_shape[0]*mask_shape[1], dtype=np.uint8)
                label = label.split(" ")
                positions = map(int, label[0::2])
                length = map(int, label[1::2])
                for pos, le in zip(positions, length):
                    mask_label[pos-1:pos+le-1] = 1
                mask = mask_label.reshape(mask_shape[0], mask_shape[1], order='F')
                contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)              
                for i in range(0, len(contours)):
                    cv2.polylines(img, contours[i], True, palet[idx], 2)
                if save_path is not None:
                    path = save_path + str(cl)
                    if not os.path.exists(path):
                        os.makedirs(path)
                    path = path + '/' +fname.split('/')[-1]
                    cv2.imwrite(path,img)
                if plot:
                    fig, ax = plt.subplots(figsize=(15, 15))
                    ax.set_title(fname)
                    ax.imshow(img)
                    plt.show()