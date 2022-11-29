# Laporan proyek machine learning – Aditya Candra Gumilang
## Project overview
Buku    merupakan    sumber    informasi    semua    aspek    kehidupan    khususnya pendidikan.  Namun  rendahnya  minat  baca  dikalangan  masyarakat  menjadi  persoalan penting    di    dunia    pendidikan    saat    ini. Bahkan berdasarkan laporan Bank Dunia, Indonesia merupakan  negara  yang  memiliki  minatbaca    sangat    rendah.    Hal    tersebut sungguh disayangkan, mengingat sebagai negara besar, Indonesia memiliki   potensi   besar   untuk   menjadi negara yang unggul.

Rendahnya minat baca dikalangan masyarakat menjadi persoalan penting didunia Pendidikan saat ini. Untuk itu diperlukan sebuah system yang dapat membantu merekomendasikan para pembaca agar lebih mudah mendapatkan informasi buku-buku yang akan dibaca selanjutnya.

## Business understanding 
#### Problem statements 
-	Bagaimana membuat system rekomendasi buku yang sesuai dengan minat calon pembaca?
-	Bagaimana membuat system rekomendasi yang menghasilkan output yang baik?
-	Bagaimana menyiapkan data untuk melatih model system rekomendasi?

#### Goals
-	Membuat system rekomendasi dengan menggunakan teknik Collaborative Filtering.
-	Membuat system rekomendasi dengan menggunakan kedua Teknik Content-based Filtering dan Collaborative Filtering.
-	Menyiapkan data dengan melakukan penanganan terhadap missing value, duplikat value, dll.

#### Solution statements
-	Content-based Filtering merupakan metode yang bekerja dengan  mencari  kedekatan  suatu  item yang   akan   direkomendasikan   ke user dengan items yang  telah  diambil  oleh pengguna sebelumnya berdasarkan kemiripan    antar    kontennya.    Namun, sistem  rekomendasi  berbasis  konten  ini masih  memiliki  kelemahan,  yaitu  karena semua informasi dipilih dan direkomendasikan   berdasarkan   konten,maka    pengguna    tidak    mendapatkan rekomendasi   pada   jenis   konten   yang berbeda.  Selain  itu,  sistem  rekomendasi ini    kurang    efektif    untuk    pengguna pemula,  karena  pengguna  yang  masih pemula   tidak   mendapat   masukan   dari pengguna sebelumnya.
<img src="https://user-images.githubusercontent.com/93992324/204537562-c419900d-db05-42bc-a00c-5df44544eaa8.png" size="500">

-	Sistem collaborative filtering adalah metode yang digunakan   untuk memprediksi kegunaan item berdasarkan penilaian pengguna sebelumnya, misalnya cara pemberian rating terhadap suatu item. Metode ini merekomendasikan item-item yang dipilih oleh   pengguna lain dengan kemiripan model item dari pengguna saat ini. Walaupun dalam     beberapa riset collaborative filtering terbukti dapat menutupi beberapa kekurangan pendekatan content  based dan  banyak diimplementasikan  dalam  aplikasi  nyata, namun pendekatan ini memiliki beberapa kekurangan, antara lain: 
    -	Cold-start  problem,  karena  pendekatan collaborative  filtering melakukan  prediksi berdasarkan rating yang  diberikan  user pada item,  maka menjadi  suatu  masalah ketika  suatu  item  baru  masuk  ke  dalam sistem  dan  belum  di-rating sama  sekali oleh  user.  Akibatnya  item  tersebut  tidak akan  pernah  direkomendasikan  kepada user.
    -	Sparsity,  untuk  ukuran  data  yang  besar, banyak item yang  baru  sedikit  di-rating oleh    user,    akibatnya item tersebut memiliki  nilai  prediksi  yang  relatif  tidak akurat   dan   menghasilkan   rekomendasi yang buruk.
<img src="https://user-images.githubusercontent.com/93992324/204538661-770558e1-7533-436f-ab4c-f50b40998f2d.png" style="width:500px;height:500px;">

## Data Understanding
Data yang saya gunakan dalam proyek ini bersumber dari situs kaggle yang dapat diakses di <a href="https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset">Book Recommendation Dataset</a>

Dataset memiliki 3 file berformat CSV (Comma-Seperated Values) di dalamnya yaitu Books.csv, Ratings.csv, dan Users.csv.

Variable yang ada pada Books.csv adalah sebagai berikut :

-	ISBN : ISBN (International Standard Book Number) merupakan id buku.
-	Book-Title : merupakan judul buku.
-	Book-Author : merupakan penulis buku.
-	Year-Of-Publication : merupakan tahun rilis buku.
-	Publisher : merupakan penerbit buku.
-	Image-URL-S : merupakan link gambar buku dalam ukuran kecil.
-	Image-URL-M : merupakan link gambar buku dalam ukuran sedang.
-	Image-URL-L : merupakan link gambar buku dalam ukuran besar.

Variable yang ada pada Ratings.csv adalah sebagai berikut :

-	User-ID : merupakan pengguna dari toko buku online.
-	Location : merupakan lokasi pengguna.
-	Age : merupakan usia pengguna.

Variable yang ada pada Users.csv adalah sebagai berikut :

-	User-ID : ID dari user yang memberikan rating terhadap buku.
-	ISBN : ISBN (International Standard Book Number) merupakan id buku.
-	Book-Rating : nilai Rating dari buku.

### Univariate Exploratory Data Analysis (EDA)

Tahap eksplorasi penting untuk memahami variabel-variabel pada data serta korelasi antar variabel. Pemahaman terhadap variabel pada data dan korelasinya akan membantu kita dalam menentukan pendekatan atau algoritma yang cocok untuk data kita.
