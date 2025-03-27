import streamlit as st
import os
from smolagents import CodeAgent, ToolCallingAgent, DuckDuckGoSearchTool, VisitWebpageTool, OpenAIServerModel

def main():
    st.title("Agent Web avec Streamlit")
    st.write("Entrez votre clé API OpenAI et votre requête pour interroger l'agent.")

    # Saisie de la clé API (en mode mot de passe)
    api_key = st.text_input("Clé API OpenAI", type="password")
    # Saisie de la requête utilisateur
    user_query = st.text_input("Requête à envoyer à l'agent")

    if st.button("Exécuter"):
        if not api_key:
            st.error("Veuillez entrer une clé API valide.")
        elif not user_query:
            st.error("Veuillez entrer une requête.")
        else:
            # Définir la clé API dans la variable d'environnement
            os.environ["OPENAI_API_KEY"] = api_key

            # Créer le modèle
            model = OpenAIServerModel(
                model_id="gpt-4o",
                api_base="https://api.openai.com/v1",
                api_key=api_key,
            )

            # Définir l'agent web avec ses outils
            web_agent = ToolCallingAgent(
                tools=[DuckDuckGoSearchTool(), VisitWebpageTool()],
                model=model,
                name="search_agent",
                description="Performs web searches using DuckDuckGo and visits webpages."
            )

            # Créer l'agent manager qui gère l'agent web
            manager_agent = CodeAgent(
                tools=[], 
                model=model, 
                name="manager_agent",
                managed_agents=[web_agent]
            )

            # Exécuter la requête et afficher le résultat
            result = manager_agent.run(user_query)
            st.subheader("Résultat de l'agent")
            st.write(result)

if __name__ == "__main__":
    main()
