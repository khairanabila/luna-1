# luna
![luna_banner](.github/luna_banner.png)
_<p align="center"> Latar belakang gambar ini dihasilkan dengan luna difusi stabil</p>_


![code_quality_checking](https://img.shields.io/github/workflow/status/slowy07/luna/CodeQL?label=Code%20quality%20check&style=flat-square)
![python_post_processing](https://img.shields.io/github/workflow/status/slowy07/luna/PythonPostPorcessing?label=Python%20Post%20Processing&style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=flat-square&logo=TensorFlow&logoColor=white)


Stable diffusion adalah pembelajaran mendalam, model teks-ke-gambar dan digunakan untuk menghasilkan gambar yang detail
dikondisikan pada deskripsi teks, hal itu juga dapat diterapkan pada tugas lain seperti inpainting atau outpainting dan
menghasilkan gambar ke panduan terjemahan gambar dengan prompt teks.

**coba dengan google collab**: 

luna dapat berjalan di notebook colab online yang dapat kalian periksa di sini:

- [luna](https://colab.research.google.com/drive/1IyHaYLCPLRurVs-tJJRLbCcO4kr1xlOi?usp=sharing)
- [luna with tpu](https://colab.research.google.com/drive/1C9xbPkt6Gysq4vFq_97S1ywGqDnqngjj?usp=sharing) 
- [luna with gpu and mixed precision](https://colab.research.google.com/drive/1ydFWPt6BgPT7i4VmZcs8ttGaFu7dw-cV?usp=sharing) 
  - Waktu pembuatan 15 detik per gambar (512 x 512) pada GPU colab default tanpa menurunkan kualitas


**penggunaan untuk colab online**:

- klik untuk menghubungkan

  ![connect](.github/connect.png)

- buka di ``runtime`` dan klik run all ( ``ctrl+f9`` jika menggunakan shortcut )
  
  ![running](.github/running.png)

- collab online menjalankan luna



# usage
### use venv
```
//create venv
python3 -m venv venv

//activate venv
source venv/bin/activate
```

### clone repo
clone on https :
```
git clone https://github.com/slowy07/luna
cd luna
pip install -r requirements.txt
```

clone on ssh :
```
git clone git@github.com:slowy07/luna.git
cd luna
pip install -r requirements.txt
```

**note** : jika menggunakan mac m1 kalian dapat mencoba menginstal ``requirements_m1.txt``

### run script
```
python text2image.py --prompt="example text"
```
untuk mengubah nama file output jalankan menggunakan `--output` flag
```
python text2image.py --prompt="cool picture" --output="cool_pic.png"
```


###  install sebagai library _Python_
```
pip install git+https://github.com/slowy07/luna
```
dan jalankan paket menggunakan
```python
from stable_diffusion_tensorflow.stable_diffusion import StableDiffusion

generator = StableDiffusion(img_height=512, img_width=512, jit_compose=False)
img = generator.generate(
  "DSLR photograph of an astronut riding a horse",
  num_steps = 50,
  unconditional_guidance_scale = 75,
  temperature = 1,
  batch_size = 1,
)
```

kalian dapat mengubah dimensi gambar dengan mengubah ``img_height`` dan ``img_width``

```python
generator = StableDiffusion(
  img_height = 1020 # or change 1080
  img_height = 1080 # or change 800
)
```

### ⚠️ NOTE ⚠️
jika pip mengalami masalah, coba jalankan pip dengan _higher privilege_ menggunakan `sudo`

# contoh

| prompt | image |
| ------ | ----- |
| minimalist house with family, mountainous forested wild, concept art illustration | ![minimalistic_house](.github/result_output/minimalist_house_with_family_mountainous_forested_wild_concept_art_illustration.png) |
| natural cave wall, dynamic light, mist low over ground, illustration by josan gonzales and moebius, studio muti, malika favre, rhads, makoto, clean thick line, comics style | ![natural_cave](.github/result_output/natural_cave.png) |
| A beautiful ultradetailed anime illustration of a city street, trending on artstation | ![anime_street_ilustration](.github/result_output/anime_street_ilustration.png) |

---
