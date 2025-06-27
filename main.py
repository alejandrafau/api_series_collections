import json
import pandas as pd
import os
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Query, HTTPException,UploadFile,File
from typing import List, Optional
from fastapi.responses import JSONResponse, FileResponse
import numpy as np
from io import StringIO
from unstack_tools import validate_descri, validate_date, get_numeric,dataframes_creator,tidy_index
import yaml


app = FastAPI(
    title="Endpoint collections",
    description="""
Prueba del endpoint collections de la API Series de tiempo
    """
)

# Permite el acceso desde cualquier dominio.
# TODO: Revisar tema de credenciales
# https://fastapi.tiangolo.com/es/tutorial/cors/?h=cors#wildcards
origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/collections/info",
         summary = "Permite consultar los atributos con los que se puede filtar una colección y los valores disponibles para estos",
         description="""Toma 3 parámetros. Hay dos obligatorios: uno es 'name', donde se indica el nombre de la colección,
         otro es 'mode', donde se indica si se quiere obtener la lista de atributos por lo que se 
         puede filtrar la colección ('attr') o si se quiere obtener los valores que puede tener
         un atributo específico ('values').
         En caso de que tenga mode 'values', también hay que proveer un valor para 'esp_attri' """)

async def get_info(
        name: str = Query(...,description="nombre de la coleccion"),
        mode: str = Query(..., pattern="^(attr|values)$", description="Modo: 'attr' o 'values'"),
        esp_attri: Optional[str] = Query(None, description="Valor opcional para filtrar")
):
    ruta = os.path.join(name,"collection_metadata.json")
    with open(ruta,"r") as f:
        collection_metadata = json.load(f)
    if mode == "values":
        if esp_attri is None:
            raise HTTPException(status_code=400, detail="Proveer atributo del que se quieren los valores disponibles")
        try:
            return JSONResponse(content=collection_metadata["general_emeta"][esp_attri])
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Atributo '{esp_attri}' no encontrado")

    return JSONResponse(content=list(collection_metadata["general_emeta"].keys()))


@app.get("/collections/data",
         summary="Permite obtener los datos de una serie de una colección mediante búsqueda paramétrica.",
         description="""Para usar este endpoint es recomendable haber consultado "/collections/info" para conocer
         atributos y valores disponibles. "collections/data" toma 3 parámetros. Uno es 'name', donde se indica el nombre de la colección,
                 otro es 'attri', donde se indica los atributos por los que interesa filtrar (Ejemplo: 'provincia_nombre',
                 'departamento_nombre','indicador'). EL tercero
                  es 'valores' con los valores que se quieren para cada atributo enlistado (Ejemplo: 'Buenos Aires',
                  'Chacabuco','superficie_sembrada_ha' """
         )
async def get_data(
        name: str = Query(...,description="nombre de la coleccion"),
        attri: List[str] = Query(..., description="Atributos de interés"),
        valores: List[str] = Query(..., description="Valores correspondientes a los atributos")
):
    # Validations
    if len(attri) != len(valores):
        raise HTTPException(status_code=400, detail="Diferencia entre cantidad de atributos y valores")

    attri_values = dict(zip(attri, valores))
    matched_series = []

    try:
        met_ruta = os.path.join(name, "collection_metadata.json")
        with open(met_ruta, "r") as f:
            collection_metadata = json.load(f)
        for distribucion, series_dict in collection_metadata['distri_emeta'].items():
            for series_id, metadata in series_dict.items():
                if all(metadata.get(attr) == val for attr, val in attri_values.items()):
                    matched_series.append((distribucion, series_id))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error buscando metadata: {e}")

    # Return appropriate response based on matches
    if len(matched_series) > 1:
        return JSONResponse(content={
            'message': "más de una serie encontrada",
            'series': [element[1] for element in matched_series]
        })

    if len(matched_series) < 1:
        return JSONResponse(content={
            'message': "ninguna serie se corresponde con los atributos",
            'series': 0
        })

    # Exactly one match
    distribucion, serie_id = matched_series[0]
    csv_path = os.path.join(name,"collections_distributions", distribucion + ".csv")

    try:
        csv_df = pd.read_csv(csv_path)
        csv_df = (csv_df
                  .rename(columns={'Unnamed: 0': 'indice_tiempo'})
                  .assign(temp_dates=lambda x: x['indice_tiempo'])
                  .set_index('temp_dates')
                  .rename_axis('indice_tiempo')
                  )
        clean_df = csv_df.replace([np.inf, -np.inf], np.nan)

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No se encontró el archivo CSV para {distribucion}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error leyendo el CSV: {e}")

    if serie_id not in csv_df.columns:
        raise HTTPException(status_code=404, detail=f"La serie '{serie_id}' no se encuentra en el archivo")


    serie = csv_df.copy()
    serie = serie[['indice_tiempo',serie_id]]
    serie = serie.replace({np.nan: None})
    serie.columns = ["indice_tiempo", "valor"]
    data = list(zip(serie["indice_tiempo"], serie["valor"]))

    return JSONResponse(content={"data": data})

@app.get("/collections/get_all",
         summary="devuelve lista con todas las colecciones disponibles")
async def get_all():
    return "Proximamente"

@app.post("/collections/unstack",
          summary = """Normaliza un csv con columnas de texto para que sea procesable por la API Series.
                    Devuelve un zip con el/los csvs normalizados y un json con atributos""")
async def unstack_sheet(
    csv_file: UploadFile = File(..., description="Archivo CSV con datos a normalizar"),
    yaml_file: UploadFile = File(..., description="Archivo YAML con los metadatos descriptivos")
):
    csv_content = await csv_file.read()
    decoded_str = csv_content.decode("ISO-8859-1")
    df = pd.read_csv(StringIO(decoded_str))
    df = df.replace(['nan', 'NaN', 'NAN', 'null', 'NULL', ''], np.nan)

    yaml_content = await yaml_file.read()
    config = yaml.safe_load(yaml_content)

    date_col = config['date_col']
    descri_cols = config['descri_cols']
    num_cols = config['num_cols']
    div_col = config['div_col']

    col_dict = {}
    col_dict = validate_date(df, date_col, col_dict)
    col_dict = validate_descri(df, descri_cols,div_col,col_dict)
    col_dict = get_numeric(df, num_cols, col_dict)
    df_index = dataframes_creator(df, col_dict)
    zip_path = tidy_index(df_index, df, col_dict)
    return FileResponse(
        path=zip_path,
        filename="collection_output.zip",
        media_type="application/zip"
    )





if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8080)