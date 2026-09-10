# main.py
import streamlit as st
import os
from graph_rag import ChronosEngine
from config import DATA_DIR

st.set_page_config(page_title="Chronos - Tutor di Storia", page_icon=":classical_building:", layout="wide")

st.title("Chronos")
st.subheader("Il tuo tutor di storia personale. Piu potente di NotebookLM.")
st.markdown("---")

def render_timeline(events):
    if not events:
        return "<p>Nessun evento trovato nei documenti.</p>"
    
    html = "<div style='position: relative; max-width: 800px; margin: 0 auto; padding: 20px 0; font-family: sans-serif;'>"
    html += "<div style='position: absolute; left: 50%; top: 0; bottom: 0; width: 4px; background: #4a90e2; transform: translateX(-50%); border-radius: 2px;'></div>"
    
    for i, ev in enumerate(events):
        side = "left" if i % 2 == 0 else "right"
        margin = "margin-right: 55%; text-align: right; padding-right: 20px;" if side == "left" else "margin-left: 55%; text-align: left; padding-left: 20px;"
        bg = "#e3f2fd" if side == "left" else "#fff3e0"
        
        html += "<div style='position: relative; margin-bottom: 30px; " + margin + "'>"
        if side == "left":
            html += "<div style='position: absolute; top: 20px; right: -8px; width: 16px; height: 16px; background: #4a90e2; border-radius: 50%; border: 3px solid white; box-shadow: 0 0 0 2px #4a90e2; z-index: 1;'></div>"
        else:
            html += "<div style='position: absolute; top: 20px; left: -8px; width: 16px; height: 16px; background: #4a90e2; border-radius: 50%; border: 3px solid white; box-shadow: 0 0 0 2px #4a90e2; z-index: 1;'></div>"
        
        html += "<div style='background: " + bg + "; padding: 15px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>"
        html += "<h4 style='margin: 0 0 5px 0; color: #1565c0; font-size: 18px;'>" + str(ev.get('anno', '???')) + "</h4>"
        html += "<h5 style='margin: 0 0 8px 0; color: #333; font-size: 15px;'>" + str(ev.get('evento', 'Evento sconosciuto')) + "</h5>"
        html += "<p style='margin: 0; font-size: 13px; color: #555; line-height: 1.4;'>" + str(ev.get('descrizione', '')) + "</p>"
        html += "</div></div>"
    
    html += "</div>"
    return html

# Inizializza il motore in sessione
if "engine" not in st.session_state:
    st.session_state.engine = ChronosEngine()
    st.session_state.ready = False

engine = st.session_state.engine

# Sidebar
with st.sidebar:
    st.header("I tuoi documenti")
    
    uploaded_file = st.file_uploader("Carica un PDF di storia", type=["pdf"])
    
    if uploaded_file is not None:
        file_path = os.path.join(DATA_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("Processa documento"):
            with st.spinner("Sto leggendo, spezzettando e costruendo il grafo di conoscenza... (puo richiedere 1-2 minuti)"):
                engine.reset()
                engine.load_pdf(file_path)
                engine.build_vectorstore()
                engine.build_knowledge_graph()
                st.session_state.ready = True
                if "timeline" in st.session_state:
                    del st.session_state.timeline
            st.success("Documento processato! Pronto per le domande.")
    
    if st.session_state.ready:
        st.markdown("---")
        st.header("Modalita")
        mode = st.radio(
            "Scegli come vuoi interagire:",
            ["Tutor (risposta diretta)", "Socrate (ragionamento)", "Narratore (roleplay)"],
            index=0
        )
        if "Narratore" in mode:
            st.session_state.mode = "narrator"
        elif "Socrate" in mode:
            st.session_state.mode = "socratic"
        else:
            st.session_state.mode = "tutor"
        
        if st.session_state.mode == "narrator":
            st.caption("Es: 'Raccontami la presa della Bastiglia come se fossi un sans-culotte'")
        
        st.markdown("---")
        st.header("Agente Storico")
        verify_active = st.checkbox("Attiva verifica fatti", value=False)
        st.session_state.verify = verify_active
        if verify_active:
            st.info("Dopo ogni risposta, un secondo AI verifichera che i fatti siano nei documenti.")
        
        st.markdown("---")
        st.header("Timeline")
        if st.button("Genera Timeline"):
            with st.spinner("Estraggo date ed eventi dal documento... (istantaneo)"):
                timeline = engine.extract_timeline()
                st.session_state.timeline = timeline
            if timeline:
                st.success("Timeline generata! " + str(len(timeline)) + " eventi trovati.")
            else:
                st.warning("Nessun evento datato trovato nel documento.")
        
        st.markdown("---")
        st.header("Grafo di Conoscenza")
        stats = engine.get_graph_stats()
        st.metric("Entita trovate", stats["nodi"])
        st.metric("Connessioni", stats["archi"])
        
        if stats["entita_top"]:
            st.markdown("**Entita centrali:**")
            for ent in stats["entita_top"][:5]:
                st.write(f"- {ent['entita']} ({ent['connessioni']} collegamenti)")

# Area principale
if not st.session_state.ready:
    st.info("Carica un PDF di storia dalla sidebar per iniziare.")
    st.markdown("""
    ### Cosa puoi chiedere:
    - "Quali sono le cause della Rivoluzione Francese?"
    - "Chi era Robespierre e che ruolo ha avuto?"
    - "Raccontami la presa della Bastiglia come se fossi un sans-culotte"
    
    **Modalita Tutor**: risposta diretta con fonti.  
    **Modalita Socrate**: ti guida al ragionamento con domande.  
    **Modalita Narratore**: ti mette nel cuore degli eventi in prima persona.
    """)
else:
    mode_label = "Narratore" if st.session_state.get("mode") == "narrator" else ("Socrate" if st.session_state.get("mode") == "socratic" else "Tutor")
    verify_label = " | Agente Storico ON" if st.session_state.get("verify") else ""
    st.success(f"Motore pronto! Modalita attiva: {mode_label}{verify_label}")
    
    # Mostra timeline se esiste
    if "timeline" in st.session_state and st.session_state.timeline:
        st.markdown("---")
        st.subheader("Timeline Storica")
        timeline_html = render_timeline(st.session_state.timeline)
        st.markdown(timeline_html, unsafe_allow_html=True)
        st.markdown("---")
    
    question = st.text_input(
        "La tua domanda:",
        placeholder="Es: Perche e scoppiata la prima guerra mondiale?"
    )
    
    if question:
        spinner_text = "Il narratore sta scrivendo la scena..."
        if st.session_state.get("mode") == "socratic":
            spinner_text = "Socrate sta pensando..."
        elif st.session_state.get("mode") == "tutor":
            spinner_text = "Sto cercando nel grafo di conoscenza..."
        
        with st.spinner(spinner_text):
            result = engine.query(question, mode=st.session_state.get("mode", "tutor"))
        
        st.markdown("### Risposta")
        st.write(result["risposta"])
        
        # Agente Storico
        if st.session_state.get("verify") and result.get("raw_docs"):
            with st.spinner("Agente Storico sta verificando i fatti..."):
                verification = engine.verify_response(result["risposta"], result["raw_docs"])
            
            if verification["status"] == "verificato":
                st.success(":white_check_mark: Agente Storico: tutti i fatti sono verificati nei documenti.")
            elif verification["status"] == "problemi":
                st.warning(":warning: Agente Storico: possibili allucinazioni rilevate!")
                for d in verification["dettagli"]:
                    st.write("- " + d)
            else:
                st.error(":x: Agente Storico: impossibile verificare.")
        
        if result.get("entita_collegate"):
            with st.expander("Entita collegate nel grafo"):
                st.write(", ".join(result["entita_collegate"]))
        
        if result.get("fonti"):
            with st.expander("Fonti usate"):
                for fonte in result["fonti"]:
                    st.markdown(f"**{fonte['source']} - Chunk {fonte['chunk']}**")
                    st.caption(fonte["preview"])