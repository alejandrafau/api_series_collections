import pandas as pd
import numpy as np
import json
import os
import tempfile
import zipfile

def validate_date(df, date_col,col_dict):
    try:
        date_col =date_col[0]
        df[date_col] = pd.to_datetime(df[date_col], format='ISO8601')
        unique_dates = df[date_col].dropna().sort_values().unique()
        inferred_freq = pd.infer_freq(unique_dates)
        if len(unique_dates) < 2:
            raise ValueError("Se necesitan al menos dos fechas")
        expected_range = pd.date_range(start=unique_dates.min(), end=unique_dates.max(), freq=inferred_freq)
        if not np.array_equal(unique_dates, expected_range):
            raise ValueError("Las fechas no son consecutivas")
        col_dict = {"date_col": date_col,"date_range":expected_range}
        return col_dict

    except Exception as e:
        return f"❌ Falló validación de fecha: {str(e)}"


def validate_descri(df, descri_cols, div_col, col_dict):
    date_col = col_dict['date_col']
    date_unique = df[date_col].nunique()

    valid_descri_cols = []

    for col in descri_cols:
        df[col] = df[col].fillna("indefinido").astype(str)
        col_unique = df[col].nunique()

        if col_unique == date_unique:
            analog_df = df[[date_col, col]].drop_duplicates()
            if col_unique == analog_df.shape[0]:
                print(f"Se eliminó '{col}' de columnas descriptivas por ser datos análogos a fecha")
                continue
        valid_descri_cols.append(col)

    if len(valid_descri_cols) < 1:
        raise ValueError("No hay columnas descriptivas válidas")

    col_dict['descri_cols'] = valid_descri_cols

    if len(valid_descri_cols) == 1:
        col_dict['div_col'] = None
        return col_dict
    else:
        div_col = div_col[0]
        if div_col != 'ninguno':
            if div_col not in valid_descri_cols:
                 col_dict['div_col'] = None
                 print(f"La variable '{div_col}' no está entre las columnas descriptivas válidas: {valid_descri_cols}")
            col_dict['div_col'] = div_col
        else:
            col_dict['div_col'] = None
        return col_dict


def get_numeric(df,num_cols,col_dict):
    for col in num_cols:
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            raise ValueError(f"La columna '{col}' no parece ser numérica")
    col_dict['num_cols'] = num_cols
    return col_dict


def independence_check(df, div_col, cols_to_check, threshold=0.1):
    """Función para verificar que una variable es independiente: engloba a otras que no la engloban a ella. De esta forma es apta para
       dividir el dataset. Devuelve True en caso de ser efectivamente independiente, False en caso de depender de otras"""

    dependencies = {}

    for col in cols_to_check:
        if col == div_col:
            continue
        grouped = df.groupby(div_col)[col].nunique()
        dep = (grouped == 1).mean()
        dependencies[col] = dep

    dep_series = pd.Series(dependencies)
    high_deps = dep_series[dep_series>threshold]
    vincu =[]
    for item,value in high_deps.items():
      if df[item].nunique()==1:
        high_deps.drop(item,inplace=True)
        continue
      vincu.append(item)
    if len(vincu)>0:
        print (f"{div_col} depende de {vincu}, no es independiente")
        return False
    else:
        print(f"{div_col} no depende de ninguna columna, es independiente")
        return True

def divisor_valid(df, col_dict,div_col):
    """Función que verifica que la variable divisora sea válida: tenga más de un valor, no tenga valores indefinidos y sea independiente. Si se
    aprueban estas condiciones devuelve True, sino False"""
    # 1. Chequear cantidad de valores únicos
    n_unicos = df[div_col].nunique()
    print(f"hay {n_unicos}en{div_col}")
    if n_unicos <= 1:
        print(f"No hay suficientes valores únicos en{div_col}")
        return False

    # 3. Verificar independencia con otras variables
    if independence_check(df, div_col,col_dict['descri_cols'], threshold=0.1) is False:
      return False

    if "indefinido" in df[div_col].tolist():
      print(f"{div_col} tiene valores indefinidos")
      return False

    print(f"{div_col} es apta para ser divisora")
    return True

def dataframes_creator(df, col_dict):
  """Función que crea un diccionario que puede tener una o varias keys. Cada key contiene a su vez key 'data' y key 'enhanced_metadata'. La key 'data' tendrá
  un dataframe con las variables dispuestas de manera longitudinal y la key 'enhanced_metadada' es un diccionario con el detalle de los atributos de cada columna del dataframe"""
  div_col = col_dict['div_col']
  date_col = col_dict['date_col']
  num_cols = col_dict['num_cols']
  descri_cols = col_dict['descri_cols']
  expected_range = col_dict['date_range']
  df_num = 1
  if div_col is None or not divisor_valid(df, col_dict,div_col):
    print("Creando un sólo dataframe")
    agrupamiento = [date_col] + descri_cols
    valores = num_cols
    grouped = df.groupby(agrupamiento)[valores].sum()
    wide_df = grouped.unstack(descri_cols)
    wide_df = wide_df.reindex(expected_range, level='date_col')
    col_mapping = {
    idx: dict(zip(wide_df.columns.names, col))
    for idx, col in enumerate(wide_df.columns)
    }
    df_name = "df" +"_"+str(df_num)
    result_dict = {}
    result_dict[df_name] = {"data": wide_df,"enhanced_metadata" : col_mapping}
    return result_dict
  else:
    print(f"Creando un dataframe por valor de {div_col}")
    result_dict = {}
    #descri_cols = [col for col in descri_cols if col != div_col]
    for value in df[div_col].unique().tolist():
      filt_df = df[df[div_col]==value]
      agrupamiento = [date_col] + descri_cols
      valores = num_cols
      grouped = filt_df.groupby(agrupamiento)[valores].sum()
      wide_df = grouped.unstack(descri_cols)
      wide_df = wide_df.reindex(expected_range, level='date_col')
      col_mapping = {
      idx: dict(zip(wide_df.columns.names, col))
      for idx, col in enumerate(wide_df.columns)
      }
      df_name = "df"+"_"+ value.strip() +"_"+str(df_num)
      result_dict[df_name] = {"data": wide_df,"enhanced_metadata" : col_mapping}
      df_num += 1
    return result_dict





def gen_collection_meta(df,col_dict):
  collection_metadata = {
        "general_emeta":{},"distri_emeta":{}}
  for col in col_dict['descri_cols']:
      valores_disp = df[col].unique().tolist()
      collection_metadata["general_emeta"][col] = valores_disp
  return collection_metadata


def tidy_index(df_index, df, col_dict):
    """
    Aplana los nombres de columnas en los dataframes generados,
    actualiza el diccionario de atributos, exporta todo a CSV y JSON temporalmente,
    y devuelve la ruta a un ZIP con todos los archivos.
    """
    collection_metadata = gen_collection_meta(df, col_dict)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Guardar CSVs y preparar metadata
        for key, data_dict in df_index.items():
            df = data_dict['data']
            metadata = data_dict['enhanced_metadata']

            new_columns = []
            for i, col in enumerate(df.columns):
                flat_col_name = '_'.join(map(str, col if isinstance(col, tuple) else [col])).strip()

                if i in metadata:
                    metadata[flat_col_name] = metadata.pop(i)
                    if None in metadata[flat_col_name]:
                        metadata[flat_col_name]['medida'] = metadata[flat_col_name].pop(None)

                new_columns.append(flat_col_name)

            df.columns = new_columns
            data_dict['data'] = df

            # Guardar CSV temporal
            csv_path = os.path.join(temp_dir, f"{key}.csv")
            df.to_csv(csv_path, index=False)
            collection_metadata["distri_emeta"][key] = metadata

        # Guardar JSON de metadata
        json_path = os.path.join(temp_dir, "collection_metadata.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(collection_metadata, f, ensure_ascii=False, indent=2)

        # Crear ZIP
        zip_path = os.path.join(tempfile.gettempdir(), "collection_output.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for filename in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, filename)
                zipf.write(file_path, arcname=filename)

    return zip_path











