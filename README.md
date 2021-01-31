# leukemia-classification
Algorithm for leukemia detection in leukocyte photos

# English

This project was developed for the final work of the Computer Vision course, taught by Prof. Dr. Lucas Ferrari de Oliveira for the Specialization Course in Applied Artificial Intelligence at the Federal University of Paraná.

### Project description:

The trabalho.zip file has leukocyte images in the central part. The images are named as "ImXXX_Y_Z.jpg". Where ImXXX is the number of the image, Y is its number of the sequence of alteration (data augmentation) and Z its class (0 or 1). Where, 0 indicates a normal patient and 1 indicates leukemia.

Using Computer Vision and/or CNNS techniques, extract characteristics from the images and make their correct classification (0 or 1). Remember to separate the training and testing groups. You can use the k-fold technique to divide the images and avoid overfitting.

### How to run the code

All the content of the code as well as the explanation of each section and what is necessary for the code to run are found either in the file <b>leukemia_en_US.ipynb </b>, or in the file <b>report.pdf</b>. The two files contain the same content, they are only on different platforms.

In order for the code to run, it will be necessary to have a Python environment such as the <a href="https://www.jetbrains.com/pt-br/pycharm/download/"> PyCharm </a> (recommended) or <a href="https://www.anaconda.com/products/individual"> Anaconda </a> generate for example. The code was made on Python 3.8, so this is the most recommended version. If necessary I can pass the PyCharm <b>venv</b> environment with everything installed.

With the Python environment installed, it will be necessary to <a href="https://jupyter.org/install">install Jupyter Notebook and/or Jupyter Labs</a>. With the notebook it is possible to follow the code step by step with its explanation, and with that the <b>Jupyter Labs is the most recommended </b>, since it is possible to follow the terminal outputs directly on the notebook. Anyway, it is possible to directly execute the file <b>leukemia.py</b> in your environment and follow the explanations in the file <b>report.pdf</b>.

With Jupyter installed, execute one of the two commands in the terminal of your Python environment to run Jupyter (on Anaconda it is possible to open directly from the list of applications if it is installed): <code> jupyter-lab </code> (recommended) or <code> jupyter notebook </code>. After running these, in the Jupyter file explorer select the file <b>leukemia_en_US.ipynb</b> to open the notebook and run the code.

<b>All the necessary images are already included in the project, performing image processing is optional.</b>

Any questions or problems with the code I am available to contact.

# Português

Este projeto foi desenvolvido para o trabalho final da discplina Visão Computacional, ministrada pelo Prof. Dr. Lucas Ferrari de Oliveira para o curso de Especialização em Inteligência Artificial Aplicada na Universidade Federal do Paraná.

### Descrição do trabalho:

A pasta trabalho.zip possui imagens de leucócitos na parte central. As imagens são nomeadas como "ImXXX_Y_Z.jpg". Onde ImXXX é o número da imagem, Y é o seu número da sequência de alteração (data augmentation) e Z a sua classe (0 ou 1). Onde, 0 indica paciente normal e 1 pacientes com leucemia.

Utilizando técnicas de Visão Computacional e/ou CNNS extraia características das imagens e faça a sua correta classificação (0 ou 1). Lembre-se de separar os grupos de treinamento e teste. Você pode utilizar a técnica de k-folds para a divisão das imagens e evitar o overfitting.

### Como rodar o código

Todo conteúdo do código bem como a explicação de cada trecho e o que é necessário para o código rodar se encontram ou no arquivo <b>leucemia_pt_BR.ipynb</b>, ou no arquivo <b>relatorio.pdf</b>. Os dois arquivos contém o mesmo conteúdo, são apenas em plataformas diferentes.

Para que o código seja executado será necessário ter um ambiente Python como os que o <a href="https://www.jetbrains.com/pt-br/pycharm/download/">PyCharm</a> (recomendado) ou o <a href="https://www.anaconda.com/products/individual">Anaconda</a> geram por exemplo. O código foi feito em cima do Python 3.8, então esta é a versão mais recomendada. Caso seja necessário eu posso passar o ambiente <b>venv</b> do PyCharm com tudo instalado. 

Já com o ambiente Python instalado será necessário <a href="https://jupyter.org/install">instalar o Jupyter Notebook e/ou Jupyter Labs</a>. Com o notebook é possível acompanhar passo a passo o código com sua explicação, sendo o <b>Jupyter Labs o mais recomendado</b>, visto que nesse é possível acompanhar as saídas do terminal direto no notebook. De qualquer forma é possível executar diretamente o arquivo <b>leukemia.py</b> no seu ambiente e acompanhar as explicações no arquivo <b>relatorio.pdf</b>.

Com o Jupyter instalado execute um dos dois comando no terminal do seu ambiente Python para executar o Jupyter (no caso do Anaconda é possível abrir direto da lista de aplicativos caso esteja instalado): <code>jupyter-lab</code> (recomendado) ou <code>jupyter notebook</code>. Após executar, no explorador de arquivos do Jupyter selecione o arquivo <b>leucemia_pt_BR.ipynb</b> para abrir o notebook e executar o código.

<b>Todas as imagens necessárias já acompanham o projeto, executar o processamento das imagens é opcional.</b>

Qualquer dúvida ou problema com o código estou a disposição.
