"""
matsim_plans.py

Generates 24-hour agent-based activity-travel diaries and standard MATSim
(Multi-Agent Transport Simulation) plans.xml for travel demand modeling,
converting predicted socio-demographics and inferred stay anchors into
executable traffic simulation inputs.

Academic References:
  - Bassolas, A., Ramasco, J. J., Herranz, R., & Cantú-Ros, O. G. (2019).
    "Mobile phone records to feed activity-based travel demand models:
    MATSim for studying a cordon toll policy in Barcelona."
    Transportation Research Part A: Policy and Practice, 121, 56–74.
    https://doi.org/10.1016/j.tra.2019.01.007
  - Hörl, S., & Balać, M. (2021).
    "Synthetic population and travel demand for Paris and Île-de-France
    based on open and public data."
    Transportation Research Part C: Emerging Technologies, 130, 103291.
    https://doi.org/10.1016/j.trc.2021.103291
  - Axhausen, K. W., & Horni, A. (2016).
    "The Multi-Agent Transport Simulation MATSim."
    Ubiquity Press.
"""

import os
import argparse
import xml.etree.ElementTree as ET
from xml.dom import minidom
import numpy as np
import pandas as pd


def _format_time_sec(seconds):
    """Formats seconds from midnight into HH:MM:SS string."""
    seconds = int(np.clip(seconds, 0, 86399))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def generate_matsim_plans(synthetic_pop_df, anchors_df, stays_df=None):
    """
    Synthesizes individual 24-hour activity-travel chains for each agent
    and produces both tabular diaries and a valid MATSim plans.xml file.
    """
    id_to_phone = dict(zip(synthetic_pop_df.index, synthetic_pop_df["phone_number"]))
    anchors_map = {}
    if anchors_df is not None:
        if "phone_number" in anchors_df.columns:
            for row in anchors_df.itertuples():
                anchors_map[str(row.phone_number)] = row
        else:
            pop_map = {}
            if os.path.exists("data/population.csv"):
                pop_ref = pd.read_csv("data/population.csv")
                if "agent_id" in pop_ref.columns and "phone_number" in pop_ref.columns:
                    pop_map = dict(zip(pop_ref["agent_id"], pop_ref["phone_number"]))
            for row in anchors_df.itertuples():
                if hasattr(row, "agent_id") and row.agent_id in pop_map:
                    anchors_map[str(pop_map[row.agent_id])] = row
                if hasattr(row, "agent_id"):
                    anchors_map[row.agent_id] = row

    diary_records = []
    root = ET.Element("plans")

    for idx, agent in synthetic_pop_df.iterrows():
        phone = agent["phone_number"]
        work_status = agent.get("predicted_work_status", "employed")
        income = agent.get("predicted_income_bracket", "lower_mid")

        anchor = anchors_map.get(str(phone), anchors_map.get(idx, None))
        h_lat = anchor.inferred_home_lat if anchor and np.isfinite(anchor.inferred_home_lat) else 12.9716
        h_lon = anchor.inferred_home_lon if anchor and np.isfinite(anchor.inferred_home_lon) else 77.5946
        w_lat = anchor.inferred_work_lat if anchor and np.isfinite(anchor.inferred_work_lat) else h_lat + 0.05
        w_lon = anchor.inferred_work_lon if anchor and np.isfinite(anchor.inferred_work_lon) else h_lon + 0.05
        has_work = bool(anchor.has_inferred_work) if anchor else (work_status in ["employed", "student"])

        # Decide primary travel mode based on income bracket (higher income -> car/rideshare)
        if income == "high":
            leg_mode = "car"
        elif income == "upper_mid":
            leg_mode = "car" if (idx % 2 == 0) else "pt"
        elif income == "lower_mid":
            leg_mode = "pt"
        else:
            leg_mode = "walk" if (idx % 3 == 0) else "pt"

        # MATSim Person XML Element
        person_elem = ET.SubElement(root, "person", id=str(phone))
        plan_elem = ET.SubElement(person_elem, "plan", selected="yes")

        # Typical Daily Schedule Synthesis
        if has_work and work_status in ["employed", "student"]:
            # Morning home departure: between 07:30 and 09:30
            dep_sec = 27000 + (hash(str(phone)) % 7200)
            work_dur_sec = 28800 + (hash(str(phone) + "dur") % 3600)  # ~8 to 9 hours
            ret_sec = dep_sec + 1800 + work_dur_sec

            # Home morning
            ET.SubElement(plan_elem, "act", type="home", x=f"{h_lon:.5f}", y=f"{h_lat:.5f}", end_time=_format_time_sec(dep_sec))
            diary_records.append({
                "phone_number": phone, "activity_seq": 1, "activity_type": "home",
                "start_time": "00:00:00", "end_time": _format_time_sec(dep_sec),
                "lat": h_lat, "lon": h_lon, "mode_to_next": leg_mode,
            })

            # Leg 1
            ET.SubElement(plan_elem, "leg", mode=leg_mode)

            # Work activity
            act_name = "work" if work_status == "employed" else "education"
            ET.SubElement(plan_elem, "act", type=act_name, x=f"{w_lon:.5f}", y=f"{w_lat:.5f}", end_time=_format_time_sec(ret_sec))
            diary_records.append({
                "phone_number": phone, "activity_seq": 2, "activity_type": act_name,
                "start_time": _format_time_sec(dep_sec + 1800), "end_time": _format_time_sec(ret_sec),
                "lat": w_lat, "lon": w_lon, "mode_to_next": leg_mode,
            })

            # Leg 2
            ET.SubElement(plan_elem, "leg", mode=leg_mode)

            # Home evening
            ET.SubElement(plan_elem, "act", type="home", x=f"{h_lon:.5f}", y=f"{h_lat:.5f}")
            diary_records.append({
                "phone_number": phone, "activity_seq": 3, "activity_type": "home",
                "start_time": _format_time_sec(ret_sec + 1800), "end_time": "23:59:59",
                "lat": h_lat, "lon": h_lon, "mode_to_next": "none",
            })
        else:
            # Non-worker / leisure day
            out_sec = 36000 + (hash(str(phone)) % 7200)
            in_sec = out_sec + 7200
            o_lat = h_lat + 0.015
            o_lon = h_lon + 0.015

            ET.SubElement(plan_elem, "act", type="home", x=f"{h_lon:.5f}", y=f"{h_lat:.5f}", end_time=_format_time_sec(out_sec))
            ET.SubElement(plan_elem, "leg", mode="walk")
            ET.SubElement(plan_elem, "act", type="leisure", x=f"{o_lon:.5f}", y=f"{o_lat:.5f}", end_time=_format_time_sec(in_sec))
            ET.SubElement(plan_elem, "leg", mode="walk")
            ET.SubElement(plan_elem, "act", type="home", x=f"{h_lon:.5f}", y=f"{h_lat:.5f}")

            diary_records.append({
                "phone_number": phone, "activity_seq": 1, "activity_type": "home",
                "start_time": "00:00:00", "end_time": _format_time_sec(out_sec),
                "lat": h_lat, "lon": h_lon, "mode_to_next": "walk",
            })
            diary_records.append({
                "phone_number": phone, "activity_seq": 2, "activity_type": "leisure",
                "start_time": _format_time_sec(out_sec + 900), "end_time": _format_time_sec(in_sec),
                "lat": o_lat, "lon": o_lon, "mode_to_next": "walk",
            })
            diary_records.append({
                "phone_number": phone, "activity_seq": 3, "activity_type": "home",
                "start_time": _format_time_sec(in_sec + 900), "end_time": "23:59:59",
                "lat": h_lat, "lon": h_lon, "mode_to_next": "none",
            })

    diaries_df = pd.DataFrame(diary_records)
    return root, diaries_df


def main():
    parser = argparse.ArgumentParser(description="Generate MATSim activity plans from population seeds.")
    parser.add_argument("--pop_file", type=str, default="output/synthetic_population_final.csv")
    parser.add_argument("--anchors_file", type=str, default="output/home_work_anchors.csv")
    parser.add_argument("--out_dir", type=str, default="output")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(args.pop_file):
        raise FileNotFoundError(f"Population file {args.pop_file} not found. Run ipu.py first.")

    print(f"Loading expanded synthetic population from {args.pop_file}...")
    pop_df = pd.read_csv(args.pop_file)

    anchors_df = None
    if os.path.exists(args.anchors_file):
        print(f"Loading inferred spatial anchors from {args.anchors_file}...")
        anchors_df = pd.read_csv(args.anchors_file)

    print("Synthesizing 24-hour activity-travel chains (Bassolas et al., 2019, TR-A)...")
    xml_root, diaries_df = generate_matsim_plans(pop_df, anchors_df)

    # Save tabular diaries
    diaries_csv = os.path.join(args.out_dir, "activity_travel_diaries.csv")
    diaries_df.to_csv(diaries_csv, index=False)
    print(f"Exported tabular activity-travel diaries: {diaries_csv} ({len(diaries_df)} activity episodes)")

    # Save MATSim XML plans
    xml_str = ET.tostring(xml_root, encoding="utf-8")
    dom = minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")

    # Add DOCTYPE header expected by MATSim
    doctype_header = '<!DOCTYPE plans SYSTEM "http://www.matsim.org/files/dtd/plans_v4.dtd">\n'
    if "<plans>" in pretty_xml:
        parts = pretty_xml.split("<plans>", 1)
        pretty_xml = parts[0] + doctype_header + "<plans>" + parts[1]

    plans_xml_path = os.path.join(args.out_dir, "plans.xml")
    with open(plans_xml_path, "w", encoding="utf-8") as f:
        f.write(pretty_xml)

    print(f"Exported valid MATSim XML plans: {plans_xml_path} ({len(pop_df)} synthetic agent plans)")


if __name__ == "__main__":
    main()
