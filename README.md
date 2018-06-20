###Paper : 
[Progressive Growing of GANs for Improved Quality, Stability, and Variation](http://arxiv.org/abs/1710.10196)  

You would find some helpful comments in some key functions, which will help you find detail instructions from the paper.

###ENV :
- OS: Win10
- Python 3.6.3
- CUDA 8.0
- Pytorch Windows-py3.6-cuda8
- PIL 4.3.0
- numpy 1.13.3

###How to use :
**Gen Image dataset**: Download the CelebA first, then run "gen_classified_images" function in train.py file.
```
if __name__ == "__main__":
    gen_classified_images(r"E:\workspace\datasets\CelebA\Img\img_align_celeba", centre_crop=True, save_to_local=True)
```
This function just resizing the original image, if you would like to test the CelebA-HQ dataset, please follow [tkarras'](https://github.com/tkarras/progressive_growing_of_gans) instructions.

**Training**: Open the train.py file again, modify and run the script：
```
if __name__ == "__main__":
    p = PGGAN(resolution=1024,            # Final Resolution.
              latent_size=512,            # Dimensionality of the latent vectors.
              criterion_type="GAN"        # "GAN" or "WGAN-GP"
              )
    p.train(r"E:\workspace\datasets\CelebA\Img\img_align_celeba_classified")
```

###Reference and Acknowledgement
- https://github.com/tkarras/progressive_growing_of_gans
- https://github.com/github-pengge/PyTorch-progressive_growing_of_gans
- https://github.com/caogang/wgan-gp
- https://github.com/pytorch/examples/tree/master/dcgan


