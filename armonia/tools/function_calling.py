"""
Clase 4 - Snippet 06: Function Calling Completo
Flujo completo: modelo solicita función → ejecutamos → modelo responde.
"""

import json
import os
import pandas as pd

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY"),
)
MODEL = "openai/gpt-oss-120b"

# --- Definición de la tool ---
tools = [
    {
        "type": "function",
        "function": {
            "name": "buscar_servicio",
            "description": "Busca servicios/tratamientos en el catálogo por categoría, subcategoria, nombre descripcion y el precio",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Término de búsqueda del servicio/tratamiento.",
                    },
                    "precio": {
                        "type": "number",
                        "description": "Precio máximo en pesos argentinos.",
                    },
                    "categoria": {
                        "type": "string",
                        "enum": ["Cuidados faciales", "Cuidados corporales", 
                                 "Tratamientos capilares", "Tratamientos de relajación",
                                 "Depilación", "Manos y pies", "Labios y cejas"],
                        "description": "Categoría del servicio/tratamiento.",
                    },
                    "subcategoria": {
                        "type": "string",
                        "enum": ["Reduccion/ Celulitis", "Flacidez/ Tonificacion", 
                                 "Estrías, exfoliaciones y blanqueamientos", 
                                 "Piernas cansadas / Retención de líquidos", 
                                 "Cuerpo completo", "Permanente", "Cera",
                                 "Esmaltado tradicional", "Cejas",
                                 "Pestañas", "Cejas/Pestañas", "Facial",
                                 "Labios"],
                        "description": "Subcategoría del servicio/tratamiento.",
                    }                   
                    
                },
                "required": ["categoria","query"],
            },
        },
    }
]


# --- Función ---
def buscar_servicios(
    query: str,
    categoria: str,
    subcategoria: str | None = None,
    precio: float | None = None,
    duracion: int | None = None,
):
    url = "https://docs.google.com/spreadsheets/d/1yuYwFQa4td5y1GxgRNiNo96H1eKZj5u-Y1_iedccE2k/export?format=csv"
    
    df = pd.read_csv(url, thousands=".")

    # 🔥 1. FILTRO BASE (SIEMPRE)
    resultados = df[df["categoria"] == categoria]

    # 🔍 2. QUERY (con seguro)
    if query:
        query_lower = query.lower()

        temp = resultados[
            resultados.apply(
                lambda row: query_lower in " ".join(map(str, row.values)).lower(),
                axis=1
            )
        ]

        if not temp.empty:
            resultados = temp

    # 📂 3. SUBCATEGORIA (con seguro)
    if subcategoria:
        temp = resultados[resultados["subcategoria"] == subcategoria]

        if not temp.empty:
            resultados = temp

    # 💰 4. PRECIO (con seguro)
    if precio is not None:
        temp = resultados[resultados["precio"] <= precio]

        if not temp.empty:
            resultados = temp

    # ⏱ 5. DURACION (con seguro)
    if duracion is not None:
        temp = resultados[resultados["duracion"] <= duracion]

        if not temp.empty:
            resultados = temp

    return resultados.to_dict(orient="records")


# === PASO 1: Enviar mensaje + tools al modelo ===
print("=== Paso 1: Enviando mensaje al modelo ===")
messages = [
    {"role": "system", "content": "Quiero que actues como una secretaria de una estetica. tu mision es recomendar y vender servicios y tratamiento personalizados. Responde de forma resumida y desestructurada sin abrumar de detalles al cliente"},
    {"role": "user", "content": input("Hazme tu consulta: ")}
]

response = client.chat.completions.create(
    model=MODEL,
    temperature=0.2,
    messages=messages,
    tools=tools,
)

mensaje_asistente = response.choices[0].message
print(f"El modelo quiere llamar una función: {mensaje_asistente.tool_calls is not None}")

print(mensaje_asistente.tool_calls)
tool_call = mensaje_asistente.tool_calls[0]
print(tool_call.function.arguments)
# === PASO 2: Ejecutar la función localmente ===
if mensaje_asistente.tool_calls:
    tool_call = mensaje_asistente.tool_calls[0]
    nombre_funcion = tool_call.function.name
    argumentos = json.loads(tool_call.function.arguments)

    print(f"\n=== Paso 2: Ejecutando '{nombre_funcion}' con args: {argumentos} ===")

    # Ejecutar la función
    resultado = buscar_servicios(**argumentos)
    print(f"Resultado: {json.dumps(resultado, ensure_ascii=False)}")

    # === PASO 3: Enviar resultado al modelo ===
    print("\n=== Paso 3: Enviando resultado al modelo ===")
    messages.append(mensaje_asistente)
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(resultado, ensure_ascii=False),
        }
    )

    response_final = client.chat.completions.create(
        model=MODEL,
        temperature=0.2,
        messages=messages,
        tools=tools,
    )

    print(f"\nRespuesta final del modelo:")
    print(response_final.choices[0].message.content)
