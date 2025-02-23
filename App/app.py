import streamlit as st
import joblib
import numpy as np
import warnings
import sys

# Ensure warnings module is properly registered
if 'warnings' not in sys.modules:
    import warnings
    sys.modules['warnings'] = warnings

# Load the model
@st.cache_resource
def load_model():
    try:
        return joblib.load('Konut_tahmin_model.pkl')
    except Exception as e:
        st.error(f"Model yüklenirken hata oluştu: {str(e)}")
        return None

# Streamlit app
st.title('Konut Fiyatı Tahmin Uygulaması')

# District dictionary
ilceler = {
    'Adalar': 0, 'Arnavutköy': 1, 'Ataşehir': 2, 'Avcılar': 3, 'Bahçelievler': 4, 
    'Bakırköy': 5, 'Bayramp': 6, 'Bağcılar': 7, 'Başakşe': 8, 'Beykoz': 9, 
    'Beylikdüzü': 10, 'Beyoğlu': 11, 'Beşiktaş': 12, 'Büyükçekmece': 13,
    'Esenler': 14, 'Esenyurt': 15, 'Eyüpsultan': 16, 'Fatih': 17, 'Gaziosmanpaşa': 18, 
    'Güngören': 19, 'Kadıköy': 20, 'Kartal': 21, 'Kağıthane': 22, 'Küçükçekmece': 23, 
    'Maltepe': 24, 'Pendik': 25, 'Sancaktepe': 26, 'Sarıyer': 27, 'Silivri': 28, 
    'Sultanbeyli': 29, 'Sultangazi': 30, 'Tuzla': 31, 'Zeytinburnu': 32, 'Çatalca': 33, 
    'Çekmeköy': 34, 'Ümraniy': 35, 'Üsküdar': 36, 'Şile': 37, 'Şişli': 38
}



mahalle_dict = {'100. Yıl Mah.': 0, '15 Temmuz Mah.': 1, '19 Mayıs Mah.': 2, '50. Yıl Mah.': 3, '6 SelvMah.': 4, '6 Çular Mah.': 5,
                '75. Yıl Mah.': 6, '9 Mayıs Mah.': 7, 'Abbasağa Mah.': 8, 'Abdurrahman Nafiz Gürman Mah.': 9, 'Abdurrahmangazi Mah.': 10, 'Acarlar Mah.': 11, 'Acıbadem Mah.': 12,
                'Adem Yavuz Mah.': 13, 'Adil Mah.': 14, 'Adnan Kahveci Mah.': 15, 'Adnan Menderes Mah.': 16, 'Ahmediye Mah.': 17, 'Ahmet Yesevi Mah.': 18, 'Akat Mah.': 19, 'Akbaba Mah.': 20, 'Akevler Mah.': 21, 'Aksaray Mah.': 22, 'Akçaburgaz Mah.': 23, 'Akören Mah.': 24, 'Akıncılar Mah.': 25, 'Akşemsettin Mah.': 26, 'Akşemsettin Mah.': 27, 'Alemdağ Mah.': 28, 'Ali Paşa Mah.': 29, 'Alibey Mah.': 30, 'Alibeyköy Mah.': 31, 'Alkent 2000 Mah.': 32, 'Altayçeşme Mah.': 33, 'Altunizade Mah.': 34, 'Altıntepe Mah.': 35, 'Altınşehir Mah.': 36, 'Ambarlı Mah.': 37, 'Anadolu Hisarı Mah.': 38, 'Anadolu Mah.': 39, 'Anadolufeneri Mah.': 40, 'Ardıçlı Mah.': 41, 'Armağanevler Mah.': 42, 'Arnavutköy Merkez Mah.': 43, 'Arnavutköy Köyü Mah.': 44, 'Asmalı Mescit Mah.': 45, 'Atakent Mah.': 46, 'Ataköy 1. Kısım Mah.': 47, 'Ataköy 2-5-6. Kısım Mah.': 48, 'Ataköy 3-4-11. Kısım Mah.': 49, 'Ataköy 7-8-9-10. Kısım Mah.': 50, 'Atalar Mah.': 51, 'Atatürk Mah.': 52, 'Ayazağa Mah.': 53, 'Aydınevler Mah.': 54, 'Aydınlar Mah.': 55, 'Aziz Mahmut Hüdayi Mah.': 56, 'Aşağı Dudullu Mah.': 57, 'Aşık Veysel Mah.': 58, 'Aşıkveysel Mah.': 59, 'Bahçeköy Kemer Mah.': 60, 'Bahçeköy Merkez Mah.': 61, 'Bahçeköy Yeni Mah.': 62, 'Bahçelievler Mah.': 63, 'Balaban Mah.': 64, 'Balat Mah.': 65, 'Balmumcu Mah.': 66, 'Baltalimanı Mah.': 67, 'Balıkyolu Mah.': 68, 'Barbaros Hayrettin Paşa Mah.': 69, 'Barbaros Hayrettin Mah.': 70, 'Barbaros Mah.': 71, 'Barış Mah.': 72, 'Basınköy Mah.': 73, 'Battalgazi Mah.': 74, 'Batı Mah.': 75, 'Bağlar Mah.': 76, 'Bağlarbaşı Mah.': 77, 'Bağlarçeşme Mah.': 78, 'Başıbüyük Mah.': 79, 'Bebek Mah.': 80, 'Bedrettin Mah.': 81, 'Bekir Sami Paşa Mah.': 82, 'Bereketzade Mah.': 83, 'Beylerbeyi Mah.': 84, 'Beylikdüzü OSB Mah.': 85, 'Beştelsiz Mah.': 86, 'Beşyol Mah.': 87, 'Birlik Mah.': 88, 'Bolluca Mah.': 89, 'Bostancı Mah.': 90, 'Bostancı Mah.': 91, 'Boyalık Mah.': 92, 'Boğazköy İstiklal Mah.': 93, 'Bulgurlu Mah.': 94, 'Burgazada Mah.': 95, 'Burhaniye Mah.': 96, 'Bülbül Mah.': 97, 'Büyük Sinekli Mah.': 98, 'Büyük Çavuşlu Mah.': 99, 'Büyükbakkalköy Mah.': 100, 'Büyükdere Mah.': 101, 'Büyükşehir Mah.': 102, 'Caddebostan Mah.': 103, 'Caferağa Mah.': 104, 'Camiikebir Mah.': 105, 'Cebeci Mah.': 106, 'Celaliye Mah.': 107, 'Cemil Meriç Mah.': 108, 'Cennet Mah.': 109, 'Cerrahpaşa Mah.': 110, 'Cevizli Mah.': 111, 'Cevizlik Mah.': 112, 'Cihangir Mah.': 113, 'Cihannüma Mah.': 114, 'Cumhuriyet Mah.': 115, 'Darüşşafaka Mah.': 116, 'Davutpaşa Mah.': 117, 'Defterdar Mah.': 118, 'Deliklikaya Mah.': 119, 'Demirkapı Mah.': 120, 'Denizköşkler Mah.': 121, 'Dereağzı Mah.': 122, 'Dikilitaş Mah.': 123, 'Dizdariye Mah.': 124, 'Doğu Mah.': 125, 'Dumlupınar Mah.': 126, 'Dursunköy Mah.': 127, 'Düğmeciler Mah.': 128, 'Ekinoba Mah.': 129, 'Ekşioğlu Mah.': 130, 'Elbasan Mah.': 131, 'Elmalı Mah.': 132, 'Elmalıkent Mah.': 133, 'Emirgân Mah.': 134, 'Emniyet Evleri Mah.': 135, 'Emniyettepe Mah.': 136, 'Erenköy Mah.': 137, 'Esatpaşa Mah.': 138, 'Esenler Mah.': 139, 'Esenevler Mah.': 140, 'Esenkent Mah.': 141, 'Esentepe Mah.': 142, 'Esenyalı Mah.': 143, 'Esenşehir Mah.': 144, 'Eski Habibler Mah.': 145, 'Etiler Mah.': 146, 'Evliya Çelebi Mah.': 147, 'Eğitim Mah.': 148, 'Fatih Mah.': 149, 'Fatih Sultan Mehmet Mah.': 150, 'Fener Mah.': 151, 'Fenerbahçe Mah.': 152, 'Feneryolu Mah.': 153, 'Ferah Mah.': 154, 'Ferahevler Mah.': 155, 'Ferhatpaşa Mah.': 156, 'Fetih Mah.': 157, 'Fetihtepe Mah.': 158, 'Fevzi Çakmak Mah.': 159, 'Fevzi Çakmak Mah.': 160, 'Feyzullah Mah.': 161, 'Fikirtepe Mah.': 162, 'Finanskent Mah.': 163, 'Firuzağa Mah.': 164, 'Firuzköy Mah.': 165, 'Fındıklı Mah.': 166, 'Gayrettepe Mah.': 167, 'Gazi Mah.': 168, 'Gençosman Mah.': 169, 'Girne Mah.': 170, 'Gökalp Mah.': 171, 'Gökevler Mah.': 172, 'Göktürk Merkez Mah.': 173, 'Göztepe Mah.': 174, 'Güllü Bağlar Mah.': 175, 'Gültepe Mah.': 176, 'Gümüşdere Mah.': 177, 'Gümüşpala Mah.': 178, 'Gümüşpınar Mah.': 179, 'Gümüşsuyu Mah.': 180, 'Gümüşyaka Mah.': 181, 'Güneşli Mah.': 182, 'Güneştepe Mah.': 183, 'Güngören Mah.': 184, 'Gürpınar Mah.': 185, 'Gürsel Mah.': 186, 'Güven Mah.': 187, 'Güzelce Mah.': 188, 'Güzeltepe Mah.': 189, 'Güzelyalı Mah.': 190, 'Güzelyurt Mah.': 191, 'Habibler Mah.': 192, 'Hacıahmet Mah.': 193, 'Hacımimi Mah.': 194, 'Hadımköy Mah.': 195, 'Halkalı Merkez Mah.': 196, 'Halıcıoğlu Mah.': 197, 'Hamidiye Mah.': 198, 'Haraççı Mah.': 199, 'Harmandere Mah.': 200, 'Harmantepe Mah.': 201, 'Hasanpaşa Mah.': 202, 'Hastane Mah.': 203, 'Havaalanı Mah.': 204, 'Haznedar Mah.': 205, 'Hekimbaşı Mah.': 206, 'Heybeliada Mah.': 207, 'Hicret Mah.': 208, 'Huzur Mah.': 209, 'Hürriyet Mah.': 210, 'Hüseyinağa Mah.': 211, 'Ihlamurkuyu Mah.': 212, 'Kadı Mehmet Efendi Mah.': 213, 'Kalyoncu Kulluğu Mah.': 214, 'Kamer Hatun Mah.': 215, 'Kamiloba Mah.': 216, 'Kanarya Mah.': 217, 'Kandilli Mah.': 218, 'Kanlıca Mah.': 219, 'Kaptanpaşa Mah.': 220, 'Karaburun Mah.': 221, 'Karadeniz Mah.': 222, 'Karadolap Mah.': 223, 'Karagümrük Mah.': 224, 'Karayolları Mah.': 225, 'Karlıbayır Mah.': 226, 'Karlıktepe Mah.': 227, 'Karlıktepe Mah.': 228, 'Kartaltepe Mah.': 229, 'Kavacık Mah.': 230, 'Kavaklı Mah.': 231, 'Kavakpınar Mah.': 232, 'Kaynarca Mah.': 233, 'Kayışdağı Mah.': 234, 'Kazlıçeşme Mah.': 235, 'Kazım Karabekir Mah.': 236, 'Kemalpaşa Mah.': 237, 'Kemer Mah.': 238, 'Keçeciler Mah.': 239, 'Kirazlı Mah.': 240, 'Kirazlıdere Mah.': 241, 'Kirazlıtepe Mah.': 242, 'Kireçburnu Mah.': 243, 'Koca Mustafa Paşa Mah.': 244, 'Kocasinan Merkez Mah.': 245, 'Kocataş Mah.': 246, 'Kocatepe Mah.': 247, 'Konaklar Mah.': 248, 'Kordonboyu Mah.': 249, 'Koza Mah.': 250, 'Kozyatağı Mah.': 251, 'Koşuyolu Mah.': 252, 'Kulaksız Mah.': 253, 'Kuleli Mah.': 254, 'Kuloğlu Mah.': 255, 'Kumburgaz Mah.': 256, 'Kumköy Mah.': 257, 'Kurtköy Mah.': 258, 'Kuruçeşme Mah.': 259, 'Kuzguncuk Mah.': 260, 'Kuşçu Mah.': 261, 'Kültür Mah.': 262, 'Küplüce Mah.': 263, 'Küçük Piyale Mah.': 264, 'Küçük Çamlıca Mah.': 265, 'Küçükbakkalköy Mah.': 266, 'Küçüksu Mah.': 267, 'Küçükyalı Mah.': 268, 'Kılıçali Paşa Mah.': 269, 'Kınalıada Mah.': 270, 'Kısıklı Mah.': 271, 'Levazım Mah.': 272, 'Levent Mah.': 273, 'Maden Mah.': 274, 'Maden Mah.': 275, 'Maden Mah.': 276, 'Mahmutbey Mah.': 277, 'Malkoçoğlu Mah.': 278, 'Maltepe Mah.': 279, 'Mareşal Fevzi Çakmak Mah.': 280, 'Mareşal Çakmak Mah.': 281, 'Marmara Mah.': 282, 'Maslak Mah.': 283, 'Mavigöl Mah.': 284, 'Mecidiye Mah.': 285, 'Mehmet Akif Ersoy Mah.': 286, 'Mehmet Akif Ersoy Mah.': 287, 'Mehmet Akif Mah.': 288, 'Mehmet Nesih Özmen Mah.': 289, 'Mehterçeşme Mah.': 290, 'Menderes Mah.': 291, 'Merdivenköy Mah.': 292, 'Merkez Mah.': 293, 'Merkezefendi Mah.': 294, 'Mevlana Mah.': 295, 'Mimar Sinan Mah.': 296, 'Mimar Sinan Merkez Mah.': 297, 'Mimaroba Mah.': 298, 'Mithatpaşa Mah.': 299, 'Muradiye Mah.': 300, 'Murat Reis Mah.': 301, 'Murat Çeşme Mah.': 302, 'Mustafa Kemal Mah.': 303, 'Müeyyetzade Mah.': 304, 'Namık Kemal Mah.': 305, 'Necip Fazıl Kısakürek Mah.': 306, 'Necip Fazıl Mah.': 307, 'Nene Hatun Mah.': 308, 'Nine Hatun Mah.': 309, 'Nisbetiye Mah.': 310, 'Nizam Mah.': 311, 'Nişancı Mah.': 312, 'Nişantepe Mah.': 313, 'Nuri Paşa Mah.': 314, 'Nurtepe Mah.': 315, 'Orhan Gazi Mah.': 316, 'Orhangazi Mah.': 317, 'Orhantepe Mah.': 318, 'Orta Mah.': 319, 'Ortabayır Mah.': 320, 'Ortaköy Mah.': 321, 'Ortaçeşme Mah.': 322, 'Oruçreis Mah.': 323, 'Osmanağa Mah.': 324, 'Osmangazi Mah.': 325, 'Osmaniye Mah.': 326, 'Ovayenice Mah.': 327, 'Parseller Mah.': 328, 'Pazariçi Mah.': 329, 'Paşabahçe Mah.': 330, 'Petroliş Mah.': 331, 'Piri Paşa Mah.': 332, 'Piri Mehmet Paşa Mah.': 333, 'Piri Reis Mah.': 334, 'Piyale Paşa Mah.': 335, 'Polonezköy Mah.': 336, 'Ptt Evleri Mah.': 337, 'Pürtelaş Hasan Efendi Mah.': 338, 'Pınar Mah.': 339, 'Pınartepe Mah.': 340, 'Rami Cuma Mah.': 341, 'Rami Yeni Mah.': 342, 'Rasimpaşa Mah.': 343, 'Reşitpaşa Mah.': 344, 'Riva Mah.': 345, 'Rumeli Hisarı Mah.': 346, 'Rumeli Kavağı Mah.': 347, 'Rüzgarlıbahçe Mah.': 348, 'Saadetdere Mah.': 349, 'Sahil Mah.': 350, 'Sahrayı Cedit Mah.': 351, 'Sakarya Mah.': 352, 'Sakızağacı Mah.': 353, 'Salacak Mah.': 354, 'Sanayi Mah.': 355, 'Sancaktepe Mah.': 356, 'Sapa Bağları Mah.': 357, 'Saray Mah.': 358, 'Sarıgöl Mah.': 359, 'Sarıyer Merkez Mah.': 360, 'Selahaddin Eyyubi Mah.': 361, 'Selam Ali Mah.': 362, 'Selimpaşa Mah.': 363, 'Selimiye Mah.': 364, 'Semizkumlar Mah.': 365, 'Seyitnizam Mah.': 366, 'Seyrantepe Mah.': 367, 'Seyyid Ömer Mah.': 368, 'Silahtarağa Mah.': 369, 'Sinanpaşa Mah.': 370, 'Sinanoba Mah.': 371, 'Site Mah.': 372, 'Siyavuşpaşa Mah.': 373, 'Soğanlı Mah.': 374, 'Soğanlık Yeni Mah.': 375, 'Soğukpınar Mah.': 376, 'Soğuksu Mah.': 377, 'Suadiye Mah.': 378, 'Sultan Murat Mah.': 379, 'Sultan Selim Mah.': 380, 'Sultaniye Mah.': 381, 'Sultantepe Mah.': 382, 'Sultançiftliği Mah.': 383, 'Sururi Mehmet Efendi Mah.': 384, 'Söğütlü Çeşme Mah.': 385, 'Süleymaniye Mah.': 386, 'Sümbül Efendi Mah.': 387, 'Sümer Mah.': 388, 'Sütlüce Mah.': 389, 'Sırapınar Mah.': 390, 'Tahtakale Mah.': 391, 'Talatpaşa Mah.': 392, 'Tantavi Mah.': 393, 'Tarabya Mah.': 394, 'Tatlısu Mah.': 395, 'Tayakadın Mah.': 396, 'Taşdelen Mah.': 397, 'Taşoluk Mah.': 398, 'Telsiz Mah.': 399, 'Telsiz Mah.': 400, 'Tepeüstü Mah.': 401, 'Tevfik Bey Mah.': 402, 'Tokatköy Mah.': 403, 'Tomtom Mah.': 404, 'Tozkoparan Mah.': 405, 'Tuna Mah.': 406, 'Turgut Reis Mah.': 407, 'Turgut Özal Mah.': 408, 'Türkbükü Mah.': 409, 'Ulus Mah.': 410, 'Uskumruköy Mah.': 411, 'Uğur Mumcu Mah.': 412, 'Valide Atik Mah.': 413, 'Velibaba Mah.': 414, 'Veliefendi Mah.': 415, 'Vişnezade Mah.': 416, 'Yahya Kahya Mah.': 417, 'Yahya Kemal Mah.': 418, 'Yakacık Yeni Mah.': 419, 'Yakacık Çarşı Mah.': 420, 'Yakuplu Mah.': 421, 'Yalı Mah.': 422, 'Yalıköy Mah.': 423, 'Yamanevler Mah.': 424, 'Yassıören Mah.': 425, 'Yavuz Selim Mah.': 426, 'Yavuztürk Mah.': 427, 'Yayalar Mah.': 428, 'Yayla Mah.': 429, 'Yeni Mah.': 430, 'Yeni Mahalle Mah.': 431, 'Yeni Sahra Mah.': 432, 'Yenibosna Merkez Mah.': 433, 'Yenidoğan Mah.': 434, 'Yenigün Mah.': 435, 'Yenikent Mah.': 436, 'Yeniköy Mah.': 437, 'Yeniköy Mah.': 438, 'Yenimahalle Mah.': 439, 'Yenişehir Mah.': 440, 'Yeni Çamlıca Mah.': 441, 'Yeşilbağlar Mah.': 442, 'Yeşilce Mah.': 443, 'Yeşilkent Mah.': 444, 'Yeşilköy Mah.': 445, 'Yeşilova Mah.': 446, 'Yeşilpınar Mah.': 447, 'Yeşiltepe Mah.': 448, 'Yeşilyurt Mah.': 449, 'Yukarı Dudullu Mah.': 450, 'Yukarı Mah.': 451, 'Yunus Emre Mah.': 452, 'Yunus Mah.': 453, 'Yıldız Mah.': 454, 'Yıldıztabya Mah.': 455, 'Yıldıztepe Mah.': 456, 'Zafer Mah.': 457, 'Zekeriyaköy Mah.': 458, 'Zeynep Kamil Mah.': 459, 'Zeytinlik Mah.': 460, 'Zuhuratbaba Mah.': 461, 'Zübeyde Hanım Mah.': 462, 'Zühtüpaşa Mah.': 463, 'Zümrütevler Mah.': 464, 'Altıntepsi Mah.': 465, 'Cevatpaşa Mah.': 466, 'Kartaltepe Mah.': 467, 'Kocatepe Mah.': 468, 'Muratpaşa Mah.': 469, 'Orta Mah.': 470, 'Terazidere Mah.': 471, 'Vatan Mah.': 472, 'Yenidoğan Mah.': 473, 'Yıldırım Mah.': 474, 'İsmetpaşa Mah.': 475, 'Acalı Mah.': 476, 'Mahmut Şevket Paşa Mah.': 477, 'Aköy Mah.': 478, 'Alaskargazi Mah.': 479, 'Alat Mah.': 480, 'Halide Edip Adıvar Mah.': 481, 'Halil Rıfat Paşa Mah.': 482, 'Arap Cami Mah.': 483, 'Apta Mah.': 484, 'Karagümrük Mah.': 485, 'Karaç İshak Mah.': 486, 'Harbiye Mah.': 487, 'Asek Sultan Mah.': 488, 'Fatih Mah.': 489, 'Yavuz Sultan Selim Mah.': 490, 'Ayıp Mah.': 491, 'Yayla Mah.': 492, 'Aşa Mah.': 493, 'Bakoz Mah.': 494, 'Cı Kasım Mah.': 495, 'Abdurrahmangazi Mah.': 496, 'Akpınar Mah.': 497, 'Atatürk Mah.': 498, 'Emek Mah.': 499, 'Eyüp Sultan Mah.': 500, 'Fatih Mah.': 501, 'Kemal Türkler Mah.': 502, 'Meclis Mah.': 503, 'Merve Mah.': 504, 'Mevlana Mah.': 505, 'Osmangazi Mah.': 506, 'Safa Mah.': 507, 'Sarıgazi Mah.': 508, 'Veysel Karani Mah.': 509, 'Yenidoğan Mah.': 510, 'Yunus Emre Mah.': 511, 'İnönü Mah.': 512, 'Mecidiyeköy Mah.': 513, 'Edikule Mah.': 514, 'Şehremini Mah.': 515, 'Şehsuvar Bey Mah.': 516, 'Tepeören Mah.': 517, 'Eriköy Mah.': 518, 'Merkez Mah.': 519, 'Cerrah Mah.': 520, 'Derviş Ali Mah.': 521, 'Mevlanakapı Mah.': 522, 'Beyazıt Mah.': 523, 'Çeyrek Mah.': 524, 'Seyyid Ömer Mah.': 525, 'Teşrutiyet Mah.': 526, 'Teşvikiye Mah.': 527, 'Çilköy Mah.': 528, 'Hiba Mah.': 529, 'İfa Mah.': 530, 'Cibali Mah.': 531, 'Mimar Hayrettin Mah.': 532, 'Mimar Sinan Mah.': 533, 'Binbirdirek Mah.': 534, 'İp Mustafa Çelebi Mah.': 535, 'İşanca Mah.': 536, 'Ağa Kapısı Mah.': 537, 'Kaf Kırat Mah.': 538, 'Aksaray Mah.': 539, 'Akşemsettin Mah.': 540, 'Alem Mah.': 541, 'Alibey Mah.': 542, 'Mbaba Mah.': 543, 'Ahmet Mah.': 544, 'Emi Sinan Mah.': 545, 'Mürlük Mah.': 546, 'Anıt Mah.': 547, 'İnönü Mah.': 548, 'Hoca Gıyasettin Mah.': 549, 'Hoca Paşa Mah.': 550, 'Hoca Mustafa Paşa Mah.': 551, 'Molla Gürani Mah.': 552, 'Molla Hüsrev Mah.': 553, 'Postane Mah.': 554, 'Ozkurt Mah.': 555, 'Top Kasım Mah.': 556, 'Altınşehir Mah.': 557, 'Bahçeşehir 1. Kısım Mah.': 558, 'Bahçeşehir 2. Kısım Mah.': 559, 'Başak Mah.': 560, 'Başakşehir Mah.': 561, 'Güvercintepe Mah.': 562, 'Kayabaşı Mah.': 563, 'Ziya Gökalp Mah.': 564, 'Şahintepe Mah.': 565, 'deresi Mah.': 566, 'deren Mah.': 567, 'derendere Mah.': 568, 'rgenekon Mah.': 569, 'Orta Mah.': 570, 'sanlı Mah.': 571, 'sentepe Mah.': 572, 'skender Mah.': 573, 'skişehir Mah.': 574, 'stasyo Mah.': 575, 'Tika Mah.': 576, 'Tuatepe Mah.': 577, 'Muhsine Hatun Mah.': 578, 'Ulya Mah.': 579, 'Cumhuriyet Mah.': 580, 'Uştepe Mah.': 581, 'va Merkez Mah.': 582, 'Evliya Çelebi Mah.': 583, 'Vuş Mah.': 584, 'Aydınlı Mah.': 585, 'Aydıntepe Mah.': 586, 'Ayvansaray Mah.': 587, 'Ayırbaşı Mah.': 588, 'İzzet Paşa Mah.': 589, 'Kızılca Mah.': 590, 'Çakmak Mah.': 591, 'Çakmaklı Mah.': 592, 'Çamlıbahçe Mah.': 593, 'Çamlık Mah.': 594, 'Çamlıtepe Mah.': 595, 'Çamçeşme Mah.': 596, 'Çatalmeşe Mah.': 597, 'Çatma Mescit Mah.': 598, 'Çavuşoğlu Mah.': 599, 'Çayırbaşı Mah.': 600, 'Çağlayan Mah.': 601, 'Çeliktepe Mah.': 602, 'Çengeldere Mah.': 603, 'Çengelköy Mah.': 604, 'Çifte Havuzlar Mah.': 605, 'Çiftlik Mah.': 606, 'Çiğdem Mah.': 607, 'Çobançeşme Mah.': 608, 'Çubuklu Mah.': 609, 'Çukur Mah.': 610, 'Çınar Mah.': 611, 'Çınardere Mah.': 612, 'Çırpıcı Mah.': 613, 'Çırçır Mah.': 614, 'Ömer Avni Mah.': 615, 'Ömer Paşa Mah.': 616, 'Örnek Mah.': 617, 'Örnekköy Mah.': 618, 'Örnektepe Mah.': 619, 'Ünalan Mah.': 620, 'Üniversite Mah.': 621, 'Üçevler Mah.': 622, 'Üçe Mah.': 623, 'Çeşme Mah.': 624, 'Gülbahar Mah.': 625, 'Süleymaniye Mah.': 626, 'Sümbül Efendi Mah.': 627, 'Küçük Ayasofya Mah.': 628, 'Oğullu Mah.': 629, 'İcadiye Mah.': 630, 'İdealtepe Mah.': 631, 'İncirköy Mah.': 632, 'İncirtepe Mah.': 633, 'İnkılap Mah.': 634, 'İnönü Mah.': 635, 'İslambey Mah.': 636, 'İslambey Mah.': 637, 'İsmet Paşa Mah.': 638, 'İstasyo Mah.': 639, 'İstiklal Mah.': 640, 'İstinye Mah.': 641, 'İzzettin Mah.': 642, 'İçerenköy Mah.': 643, 'Hırka-ı Şerif Mah.': 644, 'Şahkulu Mah.': 645, 'Şehit Muhtar Mah.': 646, 'ŞehitMah.': 647, 'ŞemsiMah.': 648, 'Şenlikköy Mah.': 649, 'ŞerifaMah.': 650, 'ŞeyhMah.': 651, 'ŞirinevMah.': 652, 'Şirintepe Mah.': 653, 'Şilvad Mah.': 654, 'Meşrutiyet Mah.': 655}



def create_input_form():
    st.write("Lütfen aşağıdaki bilgileri girin:")
    
    ilce = st.selectbox('İlçe', list(ilceler.keys()))
    mahalle = st.selectbox('Mahalle', list(mahalle_dict.keys()))
    
    col1, col2 = st.columns(2)
    with col1:
        metre_kare = st.number_input('Metre Kare', min_value=10, max_value=10000, value=100)
        oda_sayisi = st.number_input('Oda Sayısı', min_value=1, max_value=10, value=2)
    
    with col2:
        yas = st.number_input('Bina Yaşı', min_value=0, max_value=100, value=10)
        kat = st.number_input('Bulunduğu Kat', min_value=1, max_value=50, value=1)
    
    return ilce, mahalle, metre_kare, oda_sayisi, yas, kat

def make_prediction(model, features):
    try:
        input_data = np.array(features).reshape(1, -1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prediction = model.predict(input_data)
        return prediction[0]
    except Exception as e:
        st.error(f"Tahmin yapılırken hata oluştu: {str(e)}")
        return None

def main():
    model = load_model()
    if model is None:
        return
    
    ilce, mahalle, metre_kare, oda_sayisi, yas, kat = create_input_form()
    
    if st.button('Tahmin Yap'):
        # features = [ilceler[ilce], mahalle_dict[mahalle], metre_kare, oda_sayisi, yas, kat]
        
        features = [int(metre_kare), int(oda_sayisi), int(yas), int(kat), int(ilceler[ilce]), int(mahalle_dict[mahalle])]

        prediction = make_prediction(model, features)

        
        if prediction is not None:
            st.success(f'Tahmin Edilen Fiyat: {prediction:,.2f} TL')

if __name__ == "__main__":
    main()



