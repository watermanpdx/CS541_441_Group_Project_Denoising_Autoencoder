'''
=========================================================================================================
Description:    Denoising Autoencoder Analysis
Purpose:        Analysis of image denoising via autoencoder neural-net machine learning. Below content
                contains autoencoder implementation, competing methods (NLM), and performance analysis
                methods.
Python:         Version 3.9
Authors:        Andy Dang, Killian Hanson, Manisha Yadav, Tayte Waterman
Course:         CS541 Winter
Assignment:     Group Project - Autoencoder Denoising
Date:           3/16/2023
=========================================================================================================
'''
#Imports ================================================================================================
import os                               #File management for images
import random as rd                     #Facilitates randomization of training file selection                  
import cv2 as cv                        #Image handling/processing
import numpy as np                      #Support tensorflow (np.array)
import tensorflow as tf                 #Tensorflow AI library
import skimage                          #Noise injection
from matplotlib import pyplot as plt    #Plot images and metrics info
from tqdm import tqdm                   #Status bar for long-processing items

#Constants ==============================================================================================
#Image folder structure ---------------------------------------------------------------------------
TRAIN_ROOT = 'images\\train\\'
TEST_ROOT = 'images\\test\\'
VALIDATE_ROOT = 'images\\validate\\'
GROUND = 'ground\\'
NOISY = 'noisy\\'

#Functions and Classes ==================================================================================
#Non-Local Means Denoising ------------------------------------------------------------------------
def NLM(image):
    #Wrapper function on openCV NLM
    #TODO - tune parameters/config
    return cv.fastNlMeansDenoisingColored(image,None,7,7,7,13)

#CNN Autoencoder ----------------------------------------------------------------------------------
class Autoencoder:
    def __init__(self,weights_handle='model'):
        #Constructor - Initialize NN structure and compile TensorFlow model. Attempts
        #              to reload previously trained model weights if available and
        #              compatible

        self.weights_handle = weights_handle

        #NN Model structure
        input = tf.keras.Input(shape=(64,64,3)) #64 x 64 patch images
        #Encoder
        x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer='he_normal',padding="same")(input)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding="same")(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer='he_normal',padding="same")(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding="same")(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", kernel_initializer='he_normal',padding="same")(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding="same")(x)
        
        #Decoder
        x = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=2, activation="relu",kernel_initializer='he_normal', padding="same")(x)
        x = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=2, activation="relu", kernel_initializer='he_normal',padding="same")(x)
        x = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", kernel_initializer='he_normal',padding="same")(x)
        x = tf.keras.layers.Conv2D(3, (3, 3), activation="sigmoid", kernel_initializer='he_normal',padding="same")(x)

        #Compile model
        self.model = tf.keras.Model(input,x)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=tf.keras.losses.MeanSquaredError(),
            )
        
        #Reload model weights (if they exist)
        try:
            self.model.load_weights(self.weights_handle).expect_partial()
        except:
            print('> WARNING: saved model weights could not be found/restored')
        return

    def load_training_images(self,dir=TRAIN_ROOT,max_N=100):
        #Load training images to member data. Assigns noisy images as x-input and
        #ground truth images as y-label.
        #Inputs:    dir - image directory to pull from. Level above ground/noisy folders
        #           max_N - maximum (whole) images to include in training set

        #Find training images. If limited to max_N, pick random max_N samples from set
        files = os.listdir(dir + GROUND)
        if len(files) >= max_N: files = rd.sample(files, max_N)

        #Load images from file
        print('Loading and segmenting training images...')
        nonce = True
        for i in tqdm(range(len(files))):
            #Fetch ground image. Before saving, normalize and split into patch images
            #Resultant training set is set of image patches, not whole images
            ground_path = dir + GROUND + files[i]
            ground = v_normalize(get_image(ground_path))
            g_patches = gen_patches(ground,8)

            #Fetch noisy image and process. Corresponsind noisy image to ground image is
            #linked via filename. All pairs share same name w/ noisy image appending "_noise"
            #to image name
            noisy_path = dir + NOISY + files[i].replace('.jpg', '_noise.jpg')
            noisy = v_normalize(get_image(noisy_path))
            n_patches = gen_patches(noisy,8)

            #Append patch images to training sets as member data
            if nonce:
                #Set first image directly to member data, then concatenate all others
                #TODO - likely a better way to do this.
                self.train_x = n_patches
                self.train_y = g_patches
                nonce = False
            else:
                self.train_x = np.concatenate((self.train_x,n_patches))
                self.train_y = np.concatenate((self.train_y,g_patches))
        return
        
    def train(self,labeled=True):
        #Load training images from file
        self.load_training_images()

        #Traing model. Uses noisy images as x-input, and ground truth as y-label
        print("Training model:")
        if labeled:
            y_set = self.train_y
        else:
            y_set = self.train_x
        self.model.fit(
            x=self.train_x,
            y=y_set,
            epochs=10,
            batch_size=32,
            shuffle=True,
            )
        
        #Save model weights for future re-use
        self.model.save_weights(self.weights_handle)    
        return
    
    def denoise(self,image):
        #Denoise input image using trained NN

        #Normalize image and split into patches
        image = v_normalize(image)
        patches = gen_patches(image,8)
        
        #Make prediction from NN
        prediction = self.model.predict(patches,verbose=0)

        #Recombine patch images and denormalize before returning
        denoised = stitch_patches(prediction,8)
        return v_denormalize(denoised)
    
#Image Pre/Post-Processing ------------------------------------------------------------------------
def get_image(filename):
    #Wrapper function on cv.imread. Automatically converts from BGR to RGB
    image = cv.imread(filename)
    return cv.cvtColor(image, cv.COLOR_BGR2RGB)

def normalize(pixel):
    #Scale input pixel from int [0,255] to float32 [0,1] to normalize before sending to NN
    #DO NOT USE - use vectorized v_normalize(). Operates over all elements in np.array
    return np.float32(pixel/255.0)
v_normalize = np.vectorize(normalize)   #Vectorize for use over np.array

def denormalize(pixel):
    #De-scale input pixel from float32 [0,1] to int [0,255] to denormalize image from NN
    #DO NOT USE - use vectorized v_denormalize(). Operates over all elements in np.array
    return np.uint8(pixel*255)
v_denormalize = np.vectorize(denormalize)   #Vectorize for use over np.array

def gen_patches(image,n=8):
    #Split input image into np.array of sub-image patches
    #Splits into n x n sub-images. Default n=8 yeilding 64 sub-images

    #Calculate patch dimensions and initialize empty np.array
    w,h,d = image.shape
    w = w//n
    h = h//n
    patches = np.zeros((n*n,w,h,d))

    #Iterate over sub-images and store to np.array
    for i in range(n):
        for j in range(n):
            x = i*w
            y = j*h
            patches[i*n+j] = image[x:x+w, y:y+h, :]
    return patches

def stitch_patches(patches,n=8):
    #Recombine sub-images into complete image. Takes np.array with sub-images as 0th dimension
    #Recombines image from n x n sub-images. Default n=8 expects 64 sub-images

    #Calculate image dimensions and initialize image np.array
    w,h,d = patches[0].shape
    W = w*n
    H = h*n
    image = np.zeros((W,H,d), dtype='float32')

    #Iterate over sub-images and recombine to final image
    for i in range(len(patches)):
        x = i//n
        y = i%n
        image[x*w:(x+1)*w, y*h:(y+1)*h, :] = patches[i]
    return image

#Noise Injection ----------------------------------------------------------------------------------
def generate_noisy_images(directories=[TRAIN_ROOT,TEST_ROOT,VALIDATE_ROOT],mode='gaussian',var=0.01):
    #Generate noisy images from source images in target directories
    #Inputs:    directories - array of target directories (searches in GROUND sub-folder and
    #                         generate NOISY sub-folder)
    #           mode - noise generation method (of skimage.util.random_noise())
    #           var - variance control (of skimage.util.random_noise())
    #Outputs:   none - writes resultant outputs to file in supplied directories

    for root in directories:
        print('Generating noisy images in path <' + root + NOISY + '>...')
        if not (os.path.isdir(root+NOISY)):
            os.makedirs(root+NOISY) 
        files = os.listdir(root+GROUND)
        for file in tqdm(files):
            img = get_image(root+GROUND+file)
            noise_img = skimage.util.random_noise(img, mode=mode,seed=None, clip=True, var=var)
            noise_img = np.array(255*noise_img, dtype = 'uint8')
            noise_img = cv.cvtColor(noise_img, cv.COLOR_RGB2BGR)
            cv.imwrite(root+NOISY+file.replace(".jpg", "_noise.jpg" ), noise_img)
    return

#Performance Metrics ------------------------------------------------------------------------------
#TODO - create performance metrics functions to be used on denoising analysis
def MSE(A,B):
    A = cv.cvtColor(A, cv.COLOR_RGB2GRAY)
    B = cv.cvtColor(B, cv.COLOR_RGB2GRAY)
    h, w = A.shape
    diff = cv.subtract(A, B)
    err = np.sum(diff**2)
    mse = err/(float(h*w))
    return mse

def PNSR(A,B):
    return float(tf.image.psnr(A, B, 225, name=None))

def SSIM(A,B):
    return float(tf.image.ssim(A, B, max_val=255, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03))

def batch_assess(encoder,directory=TEST_ROOT,N=100,verbose=True):
    #Assess average performance of provided autoencoder model over N random images
    #Inputs:    encoder - autoencoder model to assess
    #           dorectory - root of image directory to pull ground-truth and noisy images from
    #           N - number of images to assess

    #Fetch random images
    files = os.listdir(TEST_ROOT+GROUND)
    if len(files) >= N: files = rd.sample(files, N)

    #Iterate over images and calculate performance metrics
    print('Assessing performance over sample images...')
    #results = [[noise], [nlm], [auto]]
    #          [[mse, pnsr, ssim] ... [mse, pnsr, ssim]]
    results = [[0]*3 for i in range(3)]
    for i in tqdm(range(len(files))):
        ground = get_image(TEST_ROOT+GROUND+files[i])
        noise = get_image(TEST_ROOT+NOISY+files[i].replace('.jpg','_noise.jpg'))
        f_nlm = NLM(noise)
        f_auto = encoder.denoise(noise)

        images = [noise,f_nlm,f_auto]
        for j in range(3):
            results[j][0] += MSE(ground,images[j])
            results[j][1] += PNSR(ground,images[j])
            results[j][2] += SSIM(ground,images[j])

    #Average performance over N samples
    for i in range(3):
        for j in range(3):
            results[i][j] /= N

    if verbose:
        #Display results to terminal
        print('Average performance over ' + str(N) + ' images:')
        print('\tSource noisy image (compared to ground-truth):')
        print('\t\tMSE  = ' + str(results[0][0]))
        print('\t\tPNSR = ' + str(results[0][1]))
        print('\t\tSSIM = ' + str(results[0][2]))
        print('\tNon-Local Means:')
        print('\t\tMSE  = ' + str(results[1][0]))
        print('\t\tPNSR = ' + str(results[1][1]))
        print('\t\tSSIM = ' + str(results[1][2]))
        print('\tAutoencoder:')
        print('\t\tMSE  = ' + str(results[2][0]))
        print('\t\tPNSR = ' + str(results[2][1]))
        print('\t\tSSIM = ' + str(results[2][2]))
    return results

#Display Functions --------------------------------------------------------------------------------
def display_metrics(ground,noise,nlm,auto,filename=None,encoder_name='Autoencoder'):
    #Display metrics for input image and filtered images
    #Inputs:    ground - ground-truth, no noise image
    #           noise - noisy image provided to filters
    #           nlm - image filtered by non-local means
    #           auto - image filtered by autoencoder
    #Outputs:   none - displays to user directly or saves to file

    #Set up image/label arrays for axis control
    images = [ground, noise, nlm, auto]
    labels = ['Original Image', 'Noised Image', 'NLM', 'Autoencoder']

    #Initialize plot
    plt.close()
    fig,ax = plt.subplots(1,4)
    plt.suptitle('Image Denoising: NLM vs ' + encoder_name, y=0.8)

    #Iterate over images
    for i in range(4):
        ax[i].imshow(images[i])
        ax[i].get_xaxis().set_visible(False)
        ax[i].get_yaxis().set_visible(False)

        #Compute and display metrics, unless ground-truth image
        if i != 0:
            mse = MSE(ground,images[i])
            pnsr = PNSR(ground,images[i])
            ssim = SSIM(ground,images[i])
            labels[i] += ('\n\nMSE = ' + str(round(mse,3))
                          + '\nPSNR = ' + str(round(pnsr,3))
                          + '\nSSIM = ' + str(round(ssim,3)))
            ax[i].text(0.5,-0.15, labels[i], ha="center", va='top', transform=ax[i].transAxes)
    
    #Save to file or display to user
    if filename != None:
        plt.savefig(filename)
        print('Saved \'' + filename + '\' to file')
    else:
        plt.show()
    return

def view_samples(encoder,directory=TEST_ROOT,N=5,encoder_name='Autoencoder'):
    files = os.listdir(directory+NOISY)
    files = rd.sample(files,N)
    for file in files:
        test = get_image(directory+NOISY+file)
        ground = get_image(directory+GROUND+file.replace('_noise.jpg','.jpg'))
        f_auto = encoder.denoise(test)
        f_nlm = NLM(test)
        display_metrics(ground,test,f_nlm,f_auto,encoder_name=encoder_name)
    return

        
#Main ---------------------------------------------------------------------------------------------
def main():
    #TODO - structure into final performance implemenation + analysis (with metrics). Below
    # as initial setup to invoke current content

    #Regenerate noisy images from original data. By default does this for all image sets
    #generate_noisy_images(var=0.01)

    auto_unsupervised = Autoencoder('unsupervised')
    #auto_unsupervised.train(labeled=False)   #If commented out, Autoencoder() will pull from previously trained weights

    auto_supervised = Autoencoder('supervised')
    #auto_supervised.train(labeled=True)

    '''
    #Assess performance
    print('Unsupervised Autoencoder:')
    batch_assess(auto_unsupervised,N=250)
    print('Supervised Autoencoder:')
    batch_assess(auto_supervised,N=250)
    '''
    
    #Fetch random test image and display to user
    view_samples(auto_unsupervised,encoder_name='Unsupervised Autoencoder',N=3)
    view_samples(auto_supervised,encoder_name='Supervised Autoencoder',N=3)

    return

#Execute Main ===========================================================================================
if __name__ == "__main__":
    main()