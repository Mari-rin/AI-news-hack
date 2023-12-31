Команда MaRyRyN , DS  и разработчик Мария Шушпанова

Для данного хакатона я разработала сервис-библиотеку, которая по запросу присваивает категории новостям и идентифицирует пары схожих новостей. Был использован unsupervised-подход (обучение без учителя) на основе NMF (скрытый семантический анализ с методом неотрицательной матричной факторизации) и TF-IDF векторизации. Данные методы достаточно точны и оптимальней, задействуют меньше памяти, в отличии от больших нейронных сетей, к тому же NMF является одним из самых хорошо интерпретируемых методов среди методов обучения без учителя.

Были также опробованы методы LDA, SVD, KMeans, но они показали результат хуже.

В файле utils_test_Shushpanova_MaRyRyN.ipynb показан пример работы библиотеки, в котором категоризация текстов и идентификация схожести новостей производится на основе названий новостей (title). Но данная программа может работать и с полным объемом новостного текста (text). Для этого необходимо в параметрах вызываемых функций библиотеки указать необходимый столбец датафрейма. К сожалению, у меня не  хватило ресурсов (мощности компьютера) для того, чтобы показать пример работы программы на полном датасете на новостных статьях, но при налиии достаточных ресурсов это возможно сделать (действия см.выше).

В файле AI_news_hack_title_script_topics_similarity_Shushpanova_MaRyRyN.py находится скрипт библиотеки.

В файле nmf_model_shu.pkl содержится готовая обученная модель, но даже при ее отсутствии библиотека в состоянии запустить обучение машины с нуля, даже если нет файла предобученной модели.

Библиотека категоризует файлы по следующим категориям, которые были изначально даны в условиях задачи, а именно:
Финансы, Технологии, Политика, Шоубиз, Fashion, Крипта, Путешествия/релокация, Образовательный контент, Развлечения, Общее; возможно расширение категоризации на подкатегории.

В данной библиотеке вся работа идет только с русскоязычным текстом новостей.

Другие возможности для расширения: обработка текстов на других языках, введение подкатегорий.
