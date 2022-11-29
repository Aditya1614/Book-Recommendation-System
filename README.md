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

1. Data Buku
    ```
    RangeIndex: 271360 entries, 0 to 271359
    Data columns (total 8 columns):
    
     #   Column               Non-Null Count   Dtype 
    ---  ------               --------------   ----- 
    0   ISBN                 271360 non-null  object
    1   Book-Title           271360 non-null  object
    2   Book-Author          271359 non-null  object
    3   Year-Of-Publication  271360 non-null  object
    4   Publisher            271358 non-null  object
    5   Image-URL-S          271360 non-null  object
    6   Image-URL-M          271360 non-null  object
    7   Image-URL-L          271357 non-null  object

    dtypes: object(8)
    
    memory usage: 16.6+ MB

    Jumlah Judul Buku:  242135

    Judul Buku:  ['Classical Mythology' 'Clara Callan' 'Decision in Normandy' ...
    'Lily Dale : The True Story of the Town that Talks to the Dead'
    "Republic (World's Classics)"
    "A Guided Tour of Rene Descartes' Meditations on First Philosophy with Complete Translations of the Meditations by Ronald Rubin"]
    ```
    Berdasarkan output diatas kita dapat mengetahui bahwa file Books.csv memiliki 271360 entri dan memiliki data unik sebanyak 242135 yang antara lain 'Classical Mythology' 'Clara Callan' 'Decision in Normandy' 'Lily Dale : The True Story of the Town that Talks to the Dead'  "Republic (World's Classics)".

2. Data Rating
    ```
    RangeIndex: 1149780 entries, 0 to 1149779

    Data columns (total 3 columns):

     #   Column       Non-Null Count    Dtype 
    ---  ------       --------------    ----- 
    0   User-ID      1149780 non-null  int64 
    1   ISBN         1149780 non-null  object
    2   Book-Rating  1149780 non-null  int64 
    
    dtypes: int64(2), object(1)

    memory usage: 26.3+ MB

    Jumlah ID user:  105283

    Jumlah ISBN:  340556

    Jumlah data penilaian buku:  1149780
    ```
    Berdasarkan output diatas kita dapat mengetahui bahwa file Ratings.csv memiliki 1149780 entri dengan jumlah id user sebanyak 105283, ISBN sebanyak 340556 dan data penilaian buku sebanyal 1149780.
    
3. Data User
    ```
    RangeIndex: 278858 entries, 0 to 278857
    Data columns (total 3 columns):
    #   Column    Non-Null Count   Dtype  
    ---  ------    --------------   -----  
    0   User-ID   278858 non-null  int64  
    1   Location  278858 non-null  object 
    2   Age       168096 non-null  float64
    dtypes: float64(1), int64(1), object(1)
    memory usage: 6.4+ MB
    Jumlah user:  278858
    ```
    
    Berdasarkan output diatas kita dapat mengetahui bahwa file Users.csv memiliki data sebanyak 278858 entri dan memiliki jumlah user yang sama yakni 2778858.
    
## Data Preparation
    
### Menangani missing value
Missing value terjadi ketika data dari sebuah record tidak lengkap. Missing value sangat mempengaruhi performa model machine learning. Ada 2 (dua) opsi untuk mengatasi missing value, yaitu menghilangkan data missing value atau mengganti nilai yang hilang dengan nilai lain, seperti rata-rata dari kolom tersebut (mean) atau nilai yang paling sering muncul (modus), atau nilai tengah (median). Dalam proyek ini saya akan menghilangkan atau menghapus missing value.

1. Data Buku

    |   |   |
    |---|---|
    | ISBN | 0 |
    | Book-Title | 0 |
    | Book-Author | 1 |
    | Year-Of-Publication | 0 |
    | Publisher | 2 | 
    | Image-URL-S | 0 |
    | Image-URL-M | 0 |
    | Image-URL-L | 3 |

    dtype: int64

    Dari output diatas dapat disimpulkan data Books.csv memiliki beberapa missing value yakni pada variable Book-Author terdapat missing value sebanyak 1, Publisher sebanyak 2 dan Image-URL-L sebanyak 3. Untuk menghapus missing value kita dapat menggunakan fungsi dropna() yang disediakan oleh library pandas.
    
2. Data Rating
    |   |   |
    |---|---|
    | User-ID | 0 |
    | ISBN | 0 |
    | Book-Rating | 0 |

    dtype: int64 
    
    Dari output diatas kita dapat mengetahui bahwa data Ratings.csv tidak memiliki missing value.

3. Data User

    |   |   |
    |---|---|
    | User-ID | 0 |
    | Location | 0 |
    | Age | 110762 |

    dtype: int64
    
    Dari output diatas terdapat missing value pada data Users.csv yakni pada variable Age sebanyak 110762. Jumlah ini tentu sangat banyak jika harus dihapus. Akan tetapi, mengingat kita mempunyai 278858 entri pada data Users.csv kehilangan data 110762 tidak masalah karena kita masih mempunyai 168096 entri.
    
### Menggabungkan data rating dengan data buku

Langkah selanjutnya adalah menggabungkan data rating dengan variable ISBN, Book-Title, dan Book-Author dari data buku. Tujuannya, supaya dapat mengetahui buku mana yang telah diberi rating oleh user berdasarkan ISBN-nya. 

Setelah digabungkan, data memiliki 1149780 entri dengan 5 variable yakni User-ID, ISBN, Book-Rating, Book-Title dan Book-Author. Setelah digabungkan kita harus mengecek missing value lagi pada data baru menggunakan teknik yang sama seperti sebelumnya.

|   |   |
|---|---|
| User-ID | 0 |
| ISBN | 0 |
| Book-Rating | 0 |
| Book-Title | 118651 |
| Book-Author | 118651 |

dtype: int64

Dari output diatas kita dapat mengetahui bahwa terdapat missing value pada data Book-Title dan Book-Author masing-masing jumlahnya sama yakni 118651. Kita akan menghapus missing value tersebut sehingga sekarang kita hanya memiliki total 1031129 entri saja.

### Drop duplikat values

Selanjutnya, kita hanya akan menggunakan data unik untuk dimasukkan ke dalam proses pemodelan. Oleh karena itu, kita perlu menghapus data yang duplikat dengan fungsi drop_duplicates(). Dalam hal ini, kita membuang data duplikat pada kolom ‘ISBN’. Setelah membuang data duplikat jumlah data menjadi 270145 entri saja.

### Membuat Dictionary

Sebelum membuat dictionary, kita perlu melakukan konversi data ‘ISBN’, ‘Book-Title’ dan ‘Book-Author’ menjadi list. Dalam hal ini, saya menggunakan fungsi tolist() dari library numpy.
Setelah itu kita dapat membuat dictionary untuk menentukan pasangan key-value pada data ‘ISBN’, ‘Book-Title’ dan ‘Book-Author’ yang telah kita siapkan sebelumnya.

### Mengurangi jumlah data

Karena data terlalu banyak menyebabkan resource yang dibutuhkan melebihi kapasitas yang disediakan secara gratis oleh google colab sebagai platform melatih model machine learning ini maka data harus dikurangi menjadi 10000 data saja.

## Modelling
### Content Based Filtering
#### TF-IDF
TF-IDF merupakan singkatan dari Term Frequency — Inverse Document Frequency. Ia bertujuan untuk mengukur seberapa penting suatu kata terhadap kata-kata lain dalam dokumen. TF-IDF adalah skema representasi yang umum digunakan untuk sistem pengambilan informasi dan ekstraksi dokumen yang relevan dengan kueri tertentu. Dalam proyek ini saya akan menggunakan fungsi TfidfVectorizer dari library scikit-learn pada data ‘Book-Title’.

Menghasilkan vektor tf-idf dalam bentuk matriks dengan menggunakan fungsi todense().
```
matrix([[0., 0., 0., ..., 0., 0., 0.],
        [0., 0., 0., ..., 0., 0., 0.],
        [0., 0., 0., ..., 0., 0., 0.],
        ...,
        [0., 0., 0., ..., 0., 0., 0.],
        [0., 0., 0., ..., 0., 0., 0.],
        [0., 0., 0., ..., 0., 0., 0.]])
```

#### Cosine Similarity
Selanjutnya kita akan menghitung derajat kesamaan (similarity degree) antar data ‘Book-Author’ dengan teknik cosine similarity. Di sini, kita menggunakan fungsi cosine_similarity dari library sklearn dengan output dibawah.

| book_author | Lionel Saben | Sheila Simonson | Jennifer Lauck | PHILIP PULLMAN | R. L. Stine |
|---|---|---|---|---|---|
| Paul Berman | 0.0 | 0.0 | 0.000000 | 0.000000	| 0.000000 |
| Betty Cuthbert | 0.0 | 0.0 | 0.000000 | 0.123641 | 0.000000 |
| Joel Rosenberg | 0.0 | 0.0 | 0.000000 | 0.028249 | 0.051958 |
| Mario Benedetti | 0.0 | 0.0 | 0.000000 | 0.000000 | 0.000000 |
| Dean R. Koontz | 0.0 | 0.0 | 0.000000 | 0.025695 | 0.047260 | 
| Barbara Bretton | 0.0 | 0.0 | 0.000000 | 0.000000 | 0.042102 | 
| MARK TWAIN | 0.0 | 0.0 | 0.000000 | 0.011544 | 0.021233 |
| John Grisham | 0.0 | 0.0 | 0.000000 | 0.031799 | 0.058487 |
| Michael MacDonald | 0.0 | 0.0 | 0.000000 | 0.000000	| 0.000000 |
| Liz Ireland | 0.0 | 0.0 | 0.038979 | 0.015911 | 0.069844 |

#### Mendapatkan Rekomendasi
Sebelumnya, kita telah memiliki data similarity (kesamaan) antar buku. Sekarang kita dapat menghasilkan sejumlah buku yang akan direkomendasikan kepada pengguna. Kita akan mencoba menghasilkan buku yang mirip dengan buku yang dirilis oleh Jill McCorkle. Hasilnya sebagai berikut.

|   | book_author | book_title |
|---|---|---|
| 0 | Sonja L. Conner | NEW AMERICAN DIET |
| 1 | Martin, Ph.D. Katahn | The T-Factor Diet |
| 2 | C. Wayne Callaway	| The Callaway Diet |
| 3 | Sally & Proctor, William Langendoen	| The Preconception Gender Diet |
| 4 | ROBERT C. ATKINS	Dr. Atkin's | Diet Revolution | 	

### Collaborative Filtering
Berbeda dengan Content Based Filtering, Collaborative Filtering  membutuhkan rating dari pengguna untuk mendapatkan rekomendasi buku. Dalam hal ini, kita akan menggunakan data rating yang telah kita siapkan sebelumnya dengan persiapan tambahan sebagai berikut.
-	Melakukan persiapan data untuk menyandikan (encode) fitur ‘User-ID’ dan ‘ISBN’ ke dalam indeks integer dan melakukan mapping ke dataframe.
-	Membagi data untuk training dan validasi dengan proporsi 80:20.

#### Proses training
Pada tahap ini, model menghitung skor kecocokan antara pengguna dan ISBN (kode buku) dengan teknik embedding. Pertama, kita melakukan proses embedding terhadap data user dan ISBN. Selanjutnya, lakukan operasi perkalian dot product antara embedding user dan ISBN. Selain itu, kita juga dapat menambahkan bias untuk setiap user dan ISBN. Skor kecocokan ditetapkan dalam skala [0,1] dengan fungsi aktivasi sigmoid.

Compile model dengan menggunakan Binary Crossentropy untuk menghitung loss function, Adam (Adaptive Moment Estimation) sebagai optimizer, dan root mean squared error (RMSE) sebagai metrics evaluation. 

#### Visualisasi hasil proses training
![metrik](https://user-images.githubusercontent.com/93992324/204560137-f6ca1802-a231-4ed6-9822-e1d6cf4d9cb0.png)

#### Mendapatkan rekomendasi
Sebelumnya, pengguna telah memberi rating pada beberapa bukuyang telah mereka kunjungi. Kita menggunakan rating ini untuk membuat rekomendasi buku yang mungkin cocok untuk pengguna.

Disini kita akan mencoba mendapatkan 10 rekomendasi buku untuk user 277087 dan hasilnya sebagai berikut.

```
252/252 [==============================] - 0s 1ms/step
Showing recommendations for users: 277087
===========================
Book with high ratings from user
--------------------------------
Leon Tolstoi : Guerra y Paz
--------------------------------
Top 10 book recommendation
--------------------------------
CHRISTOPHER PAUL CURTIS : The Watsons Go to Birmingham - 1963 (Yearling Newbery)
Harper Lee : To Kill a Mockingbird
Ruth Reichl : Tender at the Bone: Growing Up at the Table
Willa Cather : My Antonia (Dover Thrift Editions)
Witi Ihimaera : The Whale Rider
Bernard Goldberg : Bias: A CBS Insider Exposes How the Media Distort the News
Maeve Binchy : This Year It Will Be Different: And Other Stories
Keith Laumer : ROGUE BOLO
Jeffrey, A. Carver : Dragon Rigger
Robert C. Atkins : Dr. Atkins' New Carbohydrate Gram Counter
```
## Referensi
M. Irfan, A. D. Cahyani, H. R. Fika, "Sistem rekomendasi : buku online dengan metode collaborative filtering" Jurnal teknologi technoscientia, 7(1), 76–84, 2014. [<a href="https://ejournal.akprind.ac.id/index.php/technoscientia/article/view/612/467">Link</a>]

A. Fahrizain, "Bag of Words vs TF-IDF — Penjelasan dan Perbedaannya" Medium. 2021 [<a href="https://medium.com/data-folks-indonesia/bag-of-words-vs-tf-idf-penjelasan-dan-perbedaannya-3739f32cdc72 ">Link</a>]
