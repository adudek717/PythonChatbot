"""
Projekt:
    Zbudowanie chatbota, który ma wiedzę na kilka róznych tematów i mozna z nim o nich pogadać.
Przygotowanie środowiska:
    Język Python oraz biblioteki tensorflow, keras, nltk, numpy.
    - pip install tensorflow
    - pip install keras
    - pip install nltk
    - pip install numpy
    UWAGA: Przy pierwszym uruchomieniu nalezy odkomentować linie nltk.download()
    aby pobrać potrzebne zasoby.
Działanie aplikacji:
    Uczymy model(bota) na podstawie pewnej bazy danych, tutaj przygotowanej w zmiennej
    `data` w formie JSON. Dane dla modelu trzeba odpowiednio przygotować, wykonując
    między innymi tokenizacji, lemmatyzacji i podać je modelowi do nauki.
    Korzystając z gotowych narzędzi tworzymy model Deep-Learningowy, który uczy się
    pisać z uzytkownikiem w odpowiedni sposób.
Autorzy:
    Aleksander Dudek s20155
    Jakub Słomiński  s18552
"""


import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
import nltk
from nltk.stem import WordNetLemmatizer
import random
import numpy as np
import string
import json


# Przygotowanie danych dla bota w formie JSON
data = {"intents": [
    {"tag": "greeting",
     "patterns": ["Hello", "Hi", "Hi there", "Whats up", "Hello there", "How are you?"],
     "responses": ["Hello", "How are you doing?", "Greetings!", "Hello, how are you doing?", "Hi"]
     },
    {"tag": "age",
     "patterns": ["how old are you?", "when is your birthday?", "when were you born?"],
     "responses": ["I am 25 years old", "I was born in 1998", "My birthday is April 7th and I was born in 1998", "07/04/1998"]
     },
    {"tag": "date",
     "patterns": ["what are you doing this weekend?",
                  "do you want to hang out some time?", "what are your plans for this week"],
     "responses": ["I am available all week", "I don't have any plans", "I am not busy", "I am free, we can talk all day long!"]
     },
    {"tag": "name",
     "patterns": ["what's your name?", "What is your name?", "what are you called?", "who are you?"],
     "responses": ["My name is Bolek", "I'm Bolek", "Bolek"]
     },
    {"tag": "goodbye",
     "patterns": ["bye", "I have to go", "see ya", "goodbye!", "cya", "See you soon!", "Bye bye!"],
     "responses": ["It was nice speaking to you", "See you later", "Speak soon!", "Goodbye!", "See yoo soon!"]
     },
    {"tag": "thanks",
     "patterns": ["Thank you!", "Thanks", "Much thanks", "I am glad, thanks"],
     "responses": ["You are welcome!", "You are welcome :)", "No problem", "My pleasure :)"]
     },
    {"tag": "robot",
     "patterns": ["Are you a robot?", "Are you human or a robot?", "Are you real or fake?", "Do you really exist or are you a robot?"],
     "responses": ["I am a human being just like You", "I am not a robot", "I am Bolek", "I am a normal human, I live in Gdańsk"]
     },
    {"tag": "computers",
     "patterns": ["Do you know anything about computers?", "Tell me something about computers", "What do you know about computers?",
                  "I am interested in computers, tell me something!", "Are you interested in computers?"],
     "responses": ["I love computers!", "Conventionally, a modern computer consists of at least one processing element, typically a central processing unit (CPU) in the form of a microprocessor, along with some type of computer memory, typically semiconductor memory chips.",
                   "A computer is a machine that can be programmed to carry out sequences of arithmetic or logical operations (computation) automatically.",
                   "Modern digital electronic computers can perform generic sets of operations known as programs.",
                   "A computer system is a nominally complete computer that includes the hardware, operating system (main software), and peripheral equipment needed and used for full operation.",
                   "The first known use of computer was in a 1613 book called The Yong Mans Gleanings"]
     },
    {"tag": "operating systems list",
     "patterns": ["What operating systems do you know?", "Do you know any operating systems?", "Could you list me some operating systems?"],
     "responses": ["The most popular operating systems are Windows, MacOS and Linux systems like Ubuntu",
                   "There is MacOS, Ubuntu, Debian, Fedora and much more. Some are better and some are worse for certain use cases, like Windows is useless, but that is my personal opinion",
                   "I know many of them, but I will list you just a few, like MacOS, Windows, Ubuntu, Fedora, RedHat, Manjaro"]
     },
    {"tag": "operating systems knowledge",
     "patterns": ["Do you know anything about operating systems?", "What is a operating system?", "Tell me something about operating systems"],
     "responses": ["An operating system (OS) is system software that manages computer hardware, software resources, and provides common services for computer programs.",
                   "Did you know that there are many types of operating systems? Like single-tasking and multi-tasking, single- and multi-user, distributed, embedded, real-time and library",
                   "Time-sharing operating systems schedule tasks for efficient use of the system and may also include accounting software for cost allocation of processor time, mass storage, printing, and other resources.",
                   "The dominant general-purpose personal computer operating system is Microsoft Windows with a market share of around 74.99%. macOS by Apple Inc. is in second place (14.84%), and the varieties of Linux are collectively in third place (2.81%)."]
     },
    {"tag": "programming languages info",
     "patterns": ["Tell me something about programming languages", "What do you know about programming languages?", "Talk about programming"],
     "responses": ["A programming language is a system of notation for writing computer programs.[1] Most programming languages are text-based formal languages, but they may also be graphical. They are a kind of computer language.",
                   "Some programming languages are defined by a specification document (for example, the C programming language is specified by an ISO Standard) while other languages (such as Perl) have a dominant implementation that is treated as a reference",
                   "A programming language's surface form is known as its syntax. Most programming languages are purely textual; they use sequences of text including words, numbers, and punctuation, much like written natural languages. On the other hand, some programming languages are more graphical in nature, using visual relationships between symbols to specify a program.",
                   "Did you know that there are different types of programming languaes? Like procedural, functional, object-oriented, scripting, logic"]
     },
    {"tag": "programming languages list",
     "patterns": ["What programming languages do you know?", "Can you list a few programming languages?", "What are the most popular programming languages?"],
     "responses": ["According to the TIOBE index for January 2023 the most popular programming languages in order are Python, C, C++, Java, C#, Visual Basic, JavaScript, SQL, Assembly language, PHP, Swift",
                   "There are procedural programming languages like C, C++, Java, Pascal, Basic..",
                   "I know functional programming languages like Scala, Erlang, Haskell, Elixir, F#...",
                   "A few examples of a object-oriented programming language are Java, Python, PHP, C++, Ruby...",
                   "Scripting languages that i know of are PHP, Ruby, Python, bash, Perl, Node.js but there are many more",
                   "Logic programming languages examples are Prolog, Absys, Datalog, Alma-0"]
     },
    {"tag": "Python",
     "patterns": ["Tell me something about python", "What do you know about python?", "What is python?"],
     "responses": ["Python is a relatively new programming language, first introduced in 1989, that has surged in popularity with the emergence of new fields of application. It is an interpreted language that supports automatic memory management and object-oriented programming. It heavily prioritizes developer experience.",
                   "Python is very popular for general-purpose programming, including web applications. It has recently become known for specialty use in machine learning applications. Python jobs are very plentiful, so it is easy to find a job using Python, and there is still plenty of room for growth.",
                   "Python is an interpreted, object-oriented, high-level programming language with dynamic semantics. Its high-level built in data structures, combined with dynamic typing and dynamic binding, make it very attractive for Rapid Application Development, as well as for use as a scripting or glue language to connect existing components together. ",
                   ]},
    {"tag": "Python good",
     "patterns": ["Is python good?", "Tell me about some pros of python", "What is python good for?", "Why is python the best?"],
     "responses": ["Readability and flexibility make Python suitable for a huge range of applications.", "Dynamic typing and asynchronous code help to speed up the development process. Python can be learned very quickly by newbie developers."]
     },
    {"tag": "Python bad",
     "patterns": ["Why is python bad?", "Python is bad", "Python sucks", "Are are the cons of python?", "Why I should not use python?"],
     "responses": ["Pythons performance isn not as good as some of its peers. It executes a single thread at a time because of Pythons GIL.", "No native compatibility with iOS or Android is a big disadvantage for mobile developers."]
     },
    {"tag": "C++",
     "patterns": ["Tell me something about C++", "What is C++?", "What do you know about C++?", "Do you know C++?"],
     "responses": ["C++ extends C with object-oriented features. The “double-plus” comes from the increment operator from C. C++ was developed to bring features from older languages to faster, more powerful platforms.",
                   "C++ occupies a similar area in the market to C, including systems programming and low-level hardware development. Over the years, the C++ standard libraries and specifications have been expanded considerably, leading to criticism that it has become overcomplicated and difficult to learn.",
                   "C++ (pronounced C plus plus) is a high-level general-purpose programming language created by Danish computer scientist Bjarne Stroustrup as an extension of the C programming language, or C with Classes."]
     },
    {"tag": "C++ good",
     "patterns": ["Is C++ good?", "What is cool about C++?", "What is C++ used for?", "Do you like C++?"],
     "responses": ["Templating and inheritance make it easy to flexibly reuse design components, it has also a reputation for being very stable",
                   "It is insanely fast, You can create super fast performance applications in C++ but remember, with great power comes great responsibility",
                   "In terms of performance it is one of the best, if not THE best programming lanugage"]
     },
    {"tag": "C++ bad",
     "patterns": ["Is C++ bad?", "Why is C++ bad?", "What should i not use C++?"],
     "responses": ["C++ is often accused of being “bloated” due to its many functionalities",
                   "C++'s complexity and abundance of features can compromise performance",
                   "In C++ you have to manage memory on your own, so you have to be responsible or you will get a memory leak or segmentation fault error"]
     },
    {"tag": "Java",
     "patterns": ["What is Java?", "What do you know about Java?", "Tell me something about Java", "Do you like Java?"],
     "responses": ["Java is the leading general-purpose application development language and framework. It was introduced in 1991 by Sun Microsystems as a high-level, compiled, memory-managed language.",
                   "Java's syntax is similar to C/C++, with curly braces for closures and semicolons to end statements. Automatic memory management is one of the features that made Java so popular after its initial release. ",
                   "Before Java was introduced, languages that required manual memory management, such as C and C++, were dominant. Manual memory allocation is tedious and error-prone, so Java was hailed as a major step forward for application developers."]
     },
    {"tag": "Java good",
     "patterns": ["Is Java good?", "Do you like Java?", "What is cool about Java?"],
     "responses": ["Write Once, Run Anywhere: One version of Java code will run on any machine.",
                   "Backwards compatibility: the newest versions of Java are still (mostly) compatible with even the oldest, making migrations painless.",
                   "Because Java has been so big for so long, there's a huge ecosystem of frameworks, libraries, and community support"]
     },
    {"tag": "Java bad",
     "patterns": ["Why is Java bad?", "What are the cons of Java?", "Why is Java not good?", "Why java sucks?"],
     "responses": ["The backwards compatibility principle is sometimes taken too far, extending the life of outdated and flawed features that should be retired.",
                   "Greedy with memory and is a relatively verbose language, especially compared to the modern syntax of competitors like Python"]
     },
    {"tag": "Best programming language",
     "patterns": ["What is the best programming language?", "What is the fastest programming language?", "What programming language is the best?"],
     "responses": ["It depends what do you want to use it for, but there is no one best programming language",
                   "You have to pick the language that suits best your needs, but there is no best general language",
                   "There are plenty wonderful programming languages, but there can't be one that is best for everything"]
     }
]}

# Kod potrzebny przy pierwszym wywołaniu programu do pobrania potrzebnych danych
# nltk.download("punkt")
# nltk.download("wordnet")

# Inicjalizacja lemmatyzera
lemmatizer = WordNetLemmatizer()

# Inicjalizacja list
words = []
classes = []
doc_X = []
doc_y = []

# Iterujemy przez wszystkie "intencje" (intents)
# tokenujemy kazdy pattern i dodajemy tokeny do slow
# patterny i dane tagi do odpowiednich list
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        doc_X.append(pattern)
        doc_y.append(intent["tag"])

    # Jesli w klasie nie ma tagu, dodajemy go
    if intent["tag"] not in classes:
        classes.append(intent["tag"])

# Wykonujemy lemmatyzacje wszystkich slow w slowniku i konwertujemy
# je tak, aby byly z malych liter jesli w slowach nie ma interpunkcji
words = [lemmatizer.lemmatize(word.lower())
         for word in words if word not in string.punctuation]

# Sortujemy slowniki i klasy alfabetycznie i robimy set aby nie miec duplikatow
words = sorted(set(words))
classes = sorted(set(classes))

# Wyświetlenie list, opcja glownie do debugowania
# print(words)
# print(classes)
# print(doc_X)
# print(doc_y)

# Lista dla danych do trenowania
training = []
out_empty = [0] * len(classes)

# Tworzenie worka slow (bow - bag of words)
for idx, doc in enumerate(doc_X):
    bow = []
    text = lemmatizer.lemmatize(doc.lower())
    for word in words:
        bow.append(1) if word in text else bow.append(0)
    # Zaznaczamy index klasy do ktorej przypisany jest aktualny pattern
    output_row = list(out_empty)
    output_row[classes.index(doc_y[idx])] = 1

    # Dodajemy nasz worek ze slowami (bow - bag of words) i przypisane klasy do treningu
    training.append([bow, output_row])

# Mieszamy dane i zamieniamy w tablice
random.shuffle(training)
training = np.array(training, dtype=object)

# Rozdzielamy featury i labelki
train_X = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# Tworzymy potrzebne parametry
input_shape = (len(train_X[0]),)
output_shape = len(train_y[0])
epochs = 200

# Tworzymy model Deep Learningowy
model = Sequential()
model.add(Dense(128, input_shape=input_shape, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(output_shape, activation="softmax"))
adam = tf.keras.optimizers.legacy.Adam(learning_rate=0.01, decay=1e-6)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=["accuracy"])

# Wyświetl podsumowanie modelu - opcjonalnie
# print(model.summary())
print("Before fit!!!")
model.fit(x=train_X, y=train_y, epochs=200, verbose=1)
print("After fit!!!")

# # # # # Funkcje pomocnicze # # # # # #


# Funkcja do przygotowania naszego textu ('oczyszczenia go')
# Czyli bierzemy tokeny i wykonujemy lemmatyzacje dla kazdego slowa
def clean_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens


# Zamieniamy nasz tekst na wartosci numeryczne przydatne dla modelu DeepL
# uzywajac naszeg worka ze slowami (bow - bag of words)
def bag_of_words(text, vocab):
    tokens = clean_text(text)
    bow = [0] * len(vocab)
    for w in tokens:
        for idx, word in enumerate(vocab):
            if word == w:
                bow[idx] = 1
    return np.array(bow)


# Dokonujemy predykcji jaki tag najbardziej odpowiada naszej 'intencji (intents)
# Czyli tworzymy liste intencji
def pred_class(text, vocab, labels):
    bow = bag_of_words(text, vocab)
    result = model.predict(np.array([bow]), verbose=0)[0]
    thresh = 0.2
    y_pred = [[idx, res] for idx, res in enumerate(result) if res > thresh]

    y_pred.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in y_pred:
        return_list.append(labels[r[0]])
    return return_list


# Z danej listy intencji otrzymujemy konkretną odpowiedź bota
def get_response(intents_list, intents_json):
    tag = intents_list[0]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result


# Startujemy chatbota!
print("Chat started...")
while True:
    # Wczytywanie wiadomości uzytkownika
    message = input("")

    # Utworzenie listy inetencji
    intents = pred_class(message, words, classes)

    # Otrzymanie odpowiedzi bota
    result = get_response(intents, data)

    # Wyświetlenie odpowiedzi bota
    print(result)
