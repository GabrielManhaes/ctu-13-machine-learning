import pandas as pd
import numpy as np
import hashlib
import os
import time
import ipaddress

files = os.listdir('./ctu-13/')
scenarios_to_bot = ['Neris', 'Neris', 'Rbot', 'Rbot', 'Virut', 'Menti', 'Sogou', 'Murlo', 'Neris', 'Rbot', 'Rbot', 'NSIS.ay', 'Virut']
df_list = list()
i = 0

"""
  Concatenação dos 13 cenários do CTU-13, criando as colunas Scenario e Bot.
"""

for _file in files:
  _df = pd.read_csv('./ctu-13/' + _file, index_col="StartTime")
  scenario = int(_file.split(".")[0])
  _df['Bot'] = scenarios_to_bot[scenario-1]
  _df['Scenario'] = scenario
  df_list.append(_df)
  i += 1

df = pd.concat(df_list)

"""
  Função de Label Encoding.
"""

def label_simple(x):
    if 'Botnet' in x:
      return 1
    elif 'Background' in x:
      return 2
    else:
        return 0

"""
  Função de correção de valores hexadecimais nas colunas Sport e Dport.
"""

def correctPort(port):
  if str(int(str(port), 16)) != str(port):
      return int(str(port), 16)

  return int(str(port))

df['State'] = df['State'].astype('category')
df['Proto'] = df['Proto'].astype('category')

"""
  Imputação de valores NaN com a moda.
"""

df['Sport'].fillna(df['Sport'].mode()[0], inplace=True)
df['Dport'].fillna(df['Dport'].mode()[0], inplace=True)
df['State'].fillna(df['State'].mode()[0], inplace=True)
df['sTos'].fillna(df['sTos'].mode()[0], inplace=True)
df['dTos'].fillna(df['dTos'].mode()[0], inplace=True)

"""
  One Hot Encoding.
"""

df = pd.get_dummies(df, columns=['Proto'], drop_first=False)

"""
  Label Encoding.
"""

df['Label'] = df['Label'].apply(label_simple)

"""
  Removendo dados rotulados como Background.
"""

df.drop(df[df['Label'] == 2].index, inplace=True, axis=0)

"""
  Aplicação da correção nas portas.
"""

df['Sport'] = df['Sport'].apply(correctPort)
df['Dport'] = df['Dport'].apply(correctPort)

"""
  Mapeando valores inválidos para categorias válidas (<?> é equivalente a <->, etc.)
"""

df['Dir'] = df['Dir'].map({
    '  <->' : '<->',
    '   ->' : '->',
    '  <-' : '<-',
    '  <?>' : '<?>',
    '  who' : 'who',
    '  <?' : '<?',

})

df['Dir'] = df['Dir'].astype('category')

"""
  One Hot Encoding.
"""

df = pd.get_dummies(df, columns=['Dir'], drop_first=False)
df = pd.get_dummies(df, columns=['sTos'], drop_first=False)
df = pd.get_dummies(df, columns=['dTos'], drop_first=False)


"""
  Computando número de valores únicos em cada coluna, para ser posteriormente utilizado nas funções de entropia.
"""

dstaddr_unique = df['DstAddr'].nunique()
sport_unique = df['Sport'].nunique()
dport_unique = df['Dport'].nunique()
state_unique = df['State'].nunique()

"""
  Funções de categorização de IPs.
"""

def classify_ip(ip):
    try:
        ip_addr = ipaddress.ip_address(ip)
        if isinstance(ip_addr,ipaddress.IPv6Address):
            return 'ipv6'
        elif isinstance(ip_addr,ipaddress.IPv4Address):
            # split on .
            octs = ip_addr.exploded.split('.')
            if 0 < int(octs[0]) < 127: return 'A'
            elif 127 < int(octs[0]) < 192: return 'B'
            elif 191 < int(octs[0]) < 224: return 'C'
            else: return 'N/A'
    except ValueError:
        return 'N/A'

def type_ip(ip):
    try:
        ip_addr = ipaddress.ip_address(ip)
        if isinstance(ip_addr,ipaddress.IPv4Address):
            # split on .
            octs = ip_addr.exploded.split('.')
            if int(octs[0]) == 10: return 'private'
            elif int(octs[0]) == 172 and (16 <= int(octs[1]) <= 31): return 'private'
            elif int(octs[0]) == 192 and int(octs[1]) == 168: return 'private'
            else: return 'public'
        return '?'
    except ValueError:
        return '?'

"""
  Funções de entropia.
"""

def entropyDstAddr(df):
  return df.nunique()/dstaddr_unique

def entropySport(df):
  return df.nunique()/sport_unique

def entropyDport(df):
  return df.nunique()/dport_unique

def entropyState(df):
  return df.nunique()/state_unique

"""
  Função que calcula a porcentagem de portas 'comuns'.
"""

def common_ports(df):
  count = 0
  common = [21,22,23,25,53,80,110,111,135,139,143,443,445,993,995,1723,3306,3389,5900,8080,8443,5432,9000]
  for port in df:
    try:
      if int(port) in common:
        count += 1
    except:
      if int(port,16) in common:
        count += 1
  return count/len(df)



df.index = pd.to_datetime(df.index)

"""
  Criando coluna derivada de DstAddr, categorizando em A, B, C, N/A ou ipv6.
"""

df['category_dstaddr'] = df['DstAddr'].apply(classify_ip)
df['category_dstaddr'] = df['category_dstaddr'].astype('category')

"""
  One Hot Encoding.
"""

df = pd.get_dummies(df, columns=['category_dstaddr'], drop_first=False)

"""
  Criando coluna derivada de DstAddr, categorizando em public, private ou ?.
"""

df['type_dstaddr'] = df['DstAddr'].apply(type_ip)
df['type_dstaddr'] = df['type_dstaddr'].astype('category')

"""
  One Hot Encoding.
"""

df = pd.get_dummies(df, columns=['type_dstaddr'], drop_first=False)

df['Min Start Time'] = df.index
df['Count'] = 0

import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)

"""
  Agrupamento por janela de 5 segundos, cenário, endereço de origem e rótulo.
"""

df = df.groupby(['Scenario', pd.Grouper(level = 'StartTime', freq='5s'), 'SrcAddr', 'Label'], observed=True).agg({
    'Dur': ['mean', 'min', 'max', 'median', 'std', 'sum'],
    'TotPkts': ['mean', 'min', 'max', 'median', 'std', 'sum'],
    'SrcBytes': ['mean', 'min', 'max', 'median', 'std', 'sum'],
    'TotBytes': ['mean', 'min', 'max', 'median', 'std', 'sum'],
    'Dir_<-':'sum',
    'Dir_->':'sum',
    'Dir_<->':'sum',
    'Dir_who':'sum',
    'Dir_<?>':'sum',
    'DstAddr': entropyDstAddr,
    'category_dstaddr_A': 'sum',
    'category_dstaddr_B': 'sum',
    'category_dstaddr_C': 'sum',
    'category_dstaddr_N/A': 'sum',
    'type_dstaddr_public': 'sum',
    'type_dstaddr_private': 'sum',
    'Sport': [entropySport,common_ports],
    'Dport': [entropyDport,common_ports],
    'State': entropyState,
    'sTos_0.0': 'sum',
    'dTos_0.0': 'sum',
    'dTos_2.0': 'sum',
    'Proto_icmp': 'sum',
    'Proto_igmp': 'sum',
    'Proto_rtp': 'sum',
    'Proto_tcp': 'sum',
    'Proto_udp': 'sum',
    'Min Start Time': lambda group: group.index.to_series().sort_values().min(),
    'Count': 'count',
})


"""
  Preenchendo valores std == NaN, gerados por conta de grupos com uma só amostra.
"""

df[('Dur', 'std')] = df[('Dur', 'std')].fillna(0)
df[('SrcBytes', 'std')] = df[('SrcBytes', 'std')].fillna(0)
df[('TotBytes', 'std')] = df[('TotBytes', 'std')].fillna(0)
df[('TotPkts', 'std')] = df[('TotPkts', 'std')].fillna(0)



"""
  Trazendo de volta a coluna Label do MultiIndex.
"""

df.reset_index(level=4, inplace=True)

"""
  Renomeando as colunas.
"""

new_columns = [
    'label',
    'dur_mean',
    'dur_min',
    'dur_max',
    'dur_median',
    'dur_std',
    'dur_sum',
    'totpkts_mean',
    'totpkts_min',
    'totpkts_max',
    'totpkts_median',
    'totpkts_std',
    'totpkts_sum',
    'srcbytes_mean',
    'srcbytes_min',
    'srcbytes_max',
    'srcbytes_median',
    'srcbytes_std',
    'srcbytes_sum',
    'totbytes_mean',
    'totbytes_min',
    'totbytes_max',
    'totbytes_median',
    'totbytes_std',
    'totbytes_sum',
    'dir_<-',
    'dir_->',
    'dir_<->',
    'dir_who',
    'dir_<?>',
    'n_hosts',
    'hosts_A',
    'hosts_B',
    'hosts_C',
    'hosts_N/A',
    'hosts_public',
    'hosts_private',
    'n_sport',
    'common_sport',
    'n_dport',
    'common_dport',
    'n_state',
    'stos_0',
    'dtos_0',
    'dtos_2',
    'n_icmp',
    'n_igmp',
    'n_rtp',
    'n_tcp',
    'n_udp',
    'conn_start_time',
    'count',
]

df.columns = new_columns

"""
  Reordenando as colunas.
"""

df = df[[
    'conn_start_time',
    'count',
    'n_hosts',
    'hosts_A',
    'hosts_B',
    'hosts_C',
    'hosts_N/A',
    'hosts_public',
    'hosts_private',
    'dur_mean',
    'dur_min',
    'dur_max',
    'dur_median',
    'dur_std',
    'dur_sum',
    'totpkts_mean',
    'totpkts_min',
    'totpkts_max',
    'totpkts_median',
    'totpkts_std',
    'totpkts_sum',
    'srcbytes_mean',
    'srcbytes_min',
    'srcbytes_max',
    'srcbytes_median',
    'srcbytes_std',
    'srcbytes_sum',
    'totbytes_mean',
    'totbytes_min',
    'totbytes_max',
    'totbytes_median',
    'totbytes_std',
    'totbytes_sum',
    'dir_<-',
    'dir_->',
    'dir_<->',
    'dir_who',
    'dir_<?>',
    'n_sport',
    'common_sport',
    'n_dport',
    'common_dport',
    'n_state',
    'stos_0',
    'dtos_0',
    'dtos_2',
    'n_icmp',
    'n_igmp',
    'n_rtp',
    'n_tcp',
    'n_udp',
    'label',
]]


"""
  Ordenando as linhas geradas pelo tempo de início da primeira amostra do grupo.
"""
df = df.sort_values('conn_start_time')

"""
  Separação dos cenários em treino e teste.
"""

bots = ['Neris', 'Neris', 'Rbot', 'Rbot', 'Virut', 'Menti', 'Sogou', 'Murlo', 'Neris', 'Rbot', 'Rbot', 'NSIS.ay', 'Virut']
train = [3,4,5,7,10,11,12,13]
test = [1,2,6,8,9]

df_train = df.iloc[df.index.get_level_values('Scenario').isin(train)]
df_test = df.iloc[df.index.get_level_values('Scenario').isin(test)]

df_train.to_pickle('train.pkl')
df_test.to_pickle('test.pkl')
