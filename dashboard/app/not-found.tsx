import Link from "next/link";

export default function NotFound() {
  return <main className="message-page"><p className="eyebrow">NOT FOUND</p><h1>That run is gone.</h1><p>It may have been deleted from local storage.</p><Link className="primary-link" href="/">Return to sessions</Link></main>;
}
