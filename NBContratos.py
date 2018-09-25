import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk import stem
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.utils import shuffle
import sklearn.metrics as metrics
from sklearn.naive_bayes import MultinomialNB
import numpy as np


class Contratos():
    
    '''
    Implementa um pipeline de NLP e de modelagem de Naive-Bayes Multinomial para classificação
    de contratos a partir da descrição de seus objetos.

    df = pandas dataframe
    col_desc = nome da coluna do dataframe que contém a descrição do objeto do contrato
    col_target = nome da coluna do dataframe que contém os contratos pré-classificados/target
    prop_teste = proporção da base de teste - uma proporção de 0.3 gerará uma base de treino
                 equivalente a 70% dos dados (float que varia de zero a um)
    path = opcional: caminho para arquivo .csv ou .xlsx que contém os dados (string)
    sep = opcional: separador utilizado para parsear o arquivo .csv (string)
    encoding = opcional: encoding do arquivo .csv (string)

    self.modelo -> atributo que retorna o modelo
    self.precisao -> precisao do modelo
    self.acuracia -> acuracia do modelo
    self.f1 -> valor para a métrica f1 do modelo
    self.dados -> retorna dataframe original, acrescido da coluna com o predict do modelo para todo o dataframe
    self.tdm -> retorna a matriz de termos por documento, em formato sparse
    self.treino -> subset de treino do dataframe original
    self.teste -> subset de teste do dataframe original, acrescido da coluna com o predict do modelo

    '''
    
    def __init__(self, df, col_desc, col_target, prop_teste, path = '', sep = ';', encoding = 'utf-8'):
    
        
        if type(df) == str:
            self.path = path
            __aux_gerar_df(sep, encoding)
        else:
            self.dados = df
        
        self.desc = col_desc
        self.t = col_target
        if self.dados[self.t].isnull().sum() > 0:
            raise ValueError('A variável target possui valores nulos, favor corrigir o problema')
        self.__pipeline_nlp()
        self.__pipeline_naive_bayes(prop_teste)
        
    
    def __aux_gerar_df(self, sep, encoding):
        
        if '.csv' in self.path:
            self.dados = pd.read_csv(self.path, sep = sep, engine = 'python', encoding = encoding)
            
        elif '.xls' in self.path:
            self.dados = pd.read_excel(self.path)
            
        else:
            raise ValueError('O tipo de arquivo inputado ainda não é suportado')
            
    def __pipeline_naive_bayes(self, prop_teste):
        
        df_embaralhado = self.dados.sample(frac=1, random_state = 42).reset_index(drop=True).copy()
        split_point = self.__aux_subset_df(self.dados, prop_teste)
        cv = CountVectorizer(tokenizer = self.__bypass_tokenizer, lowercase = False)
        self.tdm = cv.fit_transform(self.dados['tokens_stem'])
        self.treino = df_embaralhado[split_point:].copy()
        self.tdm_treino = self.tdm[split_point:]
        self.teste =  df_embaralhado[:split_point].copy()
        self.tdm_teste = self.tdm[:split_point]
        self.__metricas_modelo()
        
        text_clf = MultinomialNB().fit(self.tdm, self.dados[self.t])
        self.dados['predicted_overfitted'] = text_clf.predict(self.tdm)
                
        
    def __metricas_modelo(self):
    
        text_clf = MultinomialNB().fit(self.tdm_treino, self.treino[self.t])
        predicted = text_clf.predict(self.tdm_teste)
        self.modelo = text_clf
        self.teste['predict'] = predicted
        
        self.acuracia = metrics.accuracy_score(self.teste[self.t], predicted)
        self.precisao = metrics.precision_score(self.teste[self.t], predicted)
        self.f1 = metrics.f1_score(self.teste[self.t], predicted)
        
        print('Métricas do modelo')
        print('\n')
        print(' : '.join(['Precisao', str(self.precisao)]))
        print(' : '.join(['Acuracia', str(self.acuracia)]))
        
    def __pipeline_nlp(self):
        stemmer = stem.RSLPStemmer()
        self.__dropar_desc_nula()
        self.dados[self.desc] = self.dados[self.desc].apply(lambda x: x.lower())
        self.dados[self.desc] = self.dados[self.desc].apply(self.__remover_digitos_e_pontuacao)
        self.dados[self.desc] = self.dados[self.desc].apply(self.__remover_acentos)
        self.dados['tokens'] = self.dados[self.desc].apply(word_tokenize)
        self.dados['tokens'] = self.dados['tokens'].apply(self.__filtrar_stopwords)
        self.dados['tokens_stem'] = self.dados['tokens'].apply(
            lambda x: [stemmer.stem(palavra) for palavra in x])
         
    
    def __dropar_desc_nula(self):
        
        index_sem_desc = list(self.dados[self.dados[self.desc].isnull()].index)
        
        if index_sem_desc:
            self.dados = self.dados.drop(index_sem_desc)
            self.dados.reset_index(drop = True, inplace = True)
            print('Foram dropadas os seguintes registros que não possuem descrição:')
            print('\n')
            for item in index_sem_desc:
                print(' : '.join(['Registro índice', str(item)]))
                
    def __remover_digitos_e_pontuacao(self,txt):
        
        pontuacao = list(punctuation)
        
        adicionais =['º', '¿','¾','¾','´','`','“','”','•','ª','–','ö','»','°','«','¬','®','¼','§', '½', '‘', '’', 'ø', '±', '—']
        
        for item in adicionais:
            pontuacao.append(item)

        txt_r = ''.join([char for char in txt if not char.isdigit() and char not in pontuacao])

        return txt_r
            
    def __remover_acentos(self,txt):
        
        dici_acentos = {
        'á' : 'a', 'Á' : 'A', 'â' : 'a', 'Â' : 'A', 'à' : 'a', 'À' : 'A', 'ã' : 'a', 'Ã' : 'A',
        'é' : 'e', 'É' : 'E', 'ê' : 'e', 'Ê' : 'E', 'è' : 'e', 'È' : 'E',
        'í' : 'i', 'Í' : 'I', 'î' : 'i', 'Î' : 'I', 'ì' : 'i', 'Ì' : 'I',
        'ó' : 'o', 'Ó' : 'O', 'ô' : 'o', 'Ô' : 'O', 'ò' : 'o', 'Ò' : 'O', 'õ' : 'o', 'Õ' : 'O',
        'ú' : 'u', 'Ú' : 'U', 'û' : 'u', 'Û' : 'U', 'ù' : 'u', 'Ù' : 'U', 'ü' : 'u',
        'ç' : 'c','Ç' : 'C', 'ñ' : 'n','Ñ' : 'N'
        }
    
        for key, value in dici_acentos.items():

            if key in txt:
                txt = txt.replace(key, value)

        return txt
    
    def __filtrar_stopwords(self, item):
    
        stop = stopwords.words('portuguese')

        return [w for w in item if w not in stop]

    def __bypass_tokenizer(self, tokens):
        return tokens
    
    def __aux_subset_df(self, df, prop):
        
        return int(df.shape[0]*prop)
    
    