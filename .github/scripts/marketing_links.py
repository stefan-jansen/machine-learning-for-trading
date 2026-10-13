# Generated from content-marketing/scripts/utm.py; regenerate, do not edit.
#!/usr/bin/env python3
"""Build four-field marketing links without changing their direct destinations."""

from __future__ import annotations

import argparse
import json
import re
from datetime import date, datetime
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from zoneinfo import ZoneInfo

CONTRACT = {
    "schema_version": 1,
    "source_meaning": "current audience/placement source, not original acquisition",
    "reserved_sources": ["maven"],
    "campaign_owner": "content-marketing",
    "sources": {
        "ml4trading": ["email", "page", "referral"],
        "substack": ["email", "newsletter", "note", "post", "profile"],
        "x": ["post", "thread", "comment", "profile", "dm"],
        "linkedin": ["post", "comment", "profile", "dm"],
        "github": ["readme", "referral", "profile"],
        "reddit": ["post", "comment"],
        "discord": ["post", "dm"],
        "hn": ["post", "comment"],
    },
    "products": {
        "research-to-production": {
            "maven_slug": "research-to-production",
            "campaign_prefix": "r2p",
            "evergreen_campaign": "r2p",
            "type": "cohort",
        },
        "foundations": {
            "maven_slug": "ml4t-foundations",
            "campaign_prefix": "foundations",
            "evergreen_campaign": "foundations_self_paced",
            "type": "self_paced",
        },
        "agent-engineering": {
            "maven_slug": "agent-engineering",
            "campaign_prefix": "agent_workshop",
            "evergreen_campaign": "agent_workshop",
            "type": "workshop",
        },
        "ai-agents-workshop": {
            "maven_slug": "ml4t-ai-agents",
            "campaign_prefix": "trading_workshop",
            "evergreen_campaign": "trading_workshop",
            "type": "workshop",
        },
        "coding-agent-workshop": {
            "maven_slug": "loop-engineering",
            "campaign_prefix": "coding_workshop",
            "evergreen_campaign": "coding_workshop",
            "type": "limited_test",
        },
        "ml4t_3e": {
            "type": "book",
            "evergreen_campaign": "ml4t_3e",
            "affiliate_urls": ["https://amzn.to/4eigy2F", "https://amzn.to/44FVkGq"],
            "affiliate_tag": "ml4trading-20",
        },
    },
    "lessons": {
        "166231": {"campaign": "agent_ll_20261021", "offering_id": "agent-engineering"},
        "b146f5": {"campaign": "trading_ll_20261028", "offering_id": "research-to-production"},
        "393eee": {"campaign": "agent_ll_20261111", "offering_id": "agent-engineering"},
        "bb56cd": {"campaign": "trading_ll_20261125", "offering_id": "research-to-production"},
    },
    "historical_source_aliases_for_reporting_only": {
        "website": "ml4trading",
        "instructor_email": "ml4trading",
        "instructor": None,
        "maven": "Ambiguous unless native/owned campaign evidence establishes its meaning",
    },
    "retailer_exception": "Preserve Amazon affiliate URLs; UTMs do not establish retailer sales.",
    "attribution": "Acquisition origin and Maven first-course attribution stay separate metadata.",
}
TOKEN = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*\Z")


def identify(url: str) -> dict:
    parts = urlsplit(url)
    path = parts.path.strip("/").split("/")
    if parts.hostname == "maven.com" and len(path) >= 2 and path[0] == "p":
        return {"lesson_id": path[1], **CONTRACT["lessons"].get(path[1], {})}
    if parts.hostname == "maven.com" and len(path) >= 2 and path[0] == "stefan-jansen":
        for product, info in CONTRACT["products"].items():
            if info.get("maven_slug") == path[1]:
                return {"offering_id": product}
        return {"native_maven_slug": path[1]}
    if url in CONTRACT["products"]["ml4t_3e"]["affiliate_urls"]:
        return {"product_id": "ml4t_3e"}
    return {}


def campaign_for_url(url: str, starts_at: date | datetime | None = None) -> str:
    """Use an event campaign only for a placement promoting that particular event."""
    identity = identify(url)
    if "lesson_id" in identity:
        return identity.get("campaign", "lesson_" + identity["lesson_id"])
    product = identity.get("offering_id") or identity.get("product_id")
    if product:
        info = CONTRACT["products"][product]
        if starts_at and info["type"] not in {"self_paced", "book"}:
            if isinstance(starts_at, datetime):
                if starts_at.tzinfo is None:
                    raise ValueError("Event datetime must have a time zone")
                starts_at = starts_at.astimezone(ZoneInfo("America/New_York")).date()
            return info["campaign_prefix"] + "_" + starts_at.strftime("%Y%m%d")
        return info["evergreen_campaign"]
    if "native_maven_slug" in identity:
        slug = re.sub(r"[^a-z0-9]+", "_", identity["native_maven_slug"].lower()).strip("_")
        return "offering_" + slug
    return "ml4t_resources"


def build(url: str, source: str, medium: str, campaign: str, content: str | None) -> str:
    if source in CONTRACT["reserved_sources"]:
        raise ValueError("maven is reserved for Maven-owned promotion; use ml4trading for ours")
    if source not in CONTRACT["sources"] or medium not in CONTRACT["sources"][source]:
        raise ValueError("Use a registered distribution channel and medium")
    for name, value in (("campaign", campaign), ("content", content)):
        if not value or not TOKEN.fullmatch(value):
            raise ValueError(f"{name} must be a nonempty lowercase/underscore identifier")
    parts = urlsplit(url)
    if parts.scheme != "https" or not parts.hostname or parts.username or parts.password:
        raise ValueError("Marketing destination must be a public HTTPS URL without credentials")
    query = parse_qsl(parts.query, keep_blank_values=True)
    if any(key.lower() in {"email", "token", "signature", "api_key"} for key, _ in query):
        raise ValueError("Private or signed URLs cannot be marketing destinations")
    if parts.hostname == "amzn.to" or parts.hostname.startswith("www.amazon."):
        if parts.hostname != "amzn.to" and not any(key == "tag" for key, _ in query):
            raise ValueError("Use the approved Amazon affiliate URL")
        return url
    query = [(key, value) for key, value in query if not key.lower().startswith("utm_")]
    query.extend(
        [
            ("utm_source", source),
            ("utm_medium", medium),
            ("utm_campaign", campaign),
            ("utm_content", content),
        ]
    )
    return urlunsplit(parts._replace(query=urlencode(query)))


def record(
    url: str,
    source: str,
    medium: str,
    campaign: str,
    content: str,
    delivery_platform: str | None = None,
) -> dict:
    tagged = build(url, source, medium, campaign, content)
    if source == "ml4trading" and medium == "email" and not delivery_platform:
        raise ValueError("Record the actual email delivery platform separately")
    return {
        "url": tagged,
        "placement_source": source,
        "delivery_platform": delivery_platform,
        "distribution_medium": medium,
        "campaign_id": campaign,
        "content_id": content,
        "campaign_owner": "content-marketing",
        "first_known_acquisition_origin": None,
        "maven_first_course_attribution": None,
        "campaign_linked_purchases": None,
        "retailer_utm_attribution": None,
        **identify(url),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", nargs="?")
    parser.add_argument("--source")
    parser.add_argument("--medium")
    parser.add_argument("--campaign")
    parser.add_argument("--content")
    parser.add_argument("--platform", help="Delivery platform metadata; never added to the URL")
    parser.add_argument("--record", action="store_true", help="Emit aggregate placement metadata")
    output = parser.add_mutually_exclusive_group()
    output.add_argument(
        "--export", type=Path, help="Generate a portable copy for another repository"
    )
    output.add_argument("--contract-output", type=Path, help="Generate the shared JSON contract")
    args = parser.parse_args()
    if args.export or args.contract_output:
        if args.url or any(
            (args.source, args.medium, args.campaign, args.content, args.platform, args.record)
        ):
            parser.error("Export cannot be combined with link arguments")
        destination = args.export or args.contract_output
        destination.parent.mkdir(parents=True, exist_ok=True)
        if args.export:
            source = Path(__file__).read_text()
            source = source.removeprefix(
                "# Generated from content-marketing/scripts/utm.py; regenerate, do not edit.\n"
            )
            text = (
                "# Generated from content-marketing/scripts/utm.py; regenerate, do not edit.\n"
                + source
            )
        else:
            text = json.dumps({"_generated_from": "scripts/utm.py", **CONTRACT}, indent=2) + "\n"
        destination.write_text(text)
        return
    if not args.url or not all((args.source, args.medium, args.campaign, args.content)):
        parser.error("URL and all four tag fields are required")
    try:
        if args.record:
            print(
                json.dumps(
                    record(
                        args.url,
                        args.source,
                        args.medium,
                        args.campaign,
                        args.content,
                        args.platform,
                    )
                )
            )
        else:
            print(build(args.url, args.source, args.medium, args.campaign, args.content))
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
