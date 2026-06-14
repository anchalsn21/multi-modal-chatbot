import { MapPin, Clock, CalendarDays, Building2, BookOpen, Mic } from "lucide-react";

const features = [
  {
    icon: MapPin,
    title: "Find Locations",
    description:
      "Instantly locate any building, facility, or service on campus. Get directions and nearest entrance information.",
    iconColor: "text-indigo-600",
    iconBg: "bg-indigo-50",
    border: "border-indigo-100",
  },
  {
    icon: Clock,
    title: "Check Opening Hours",
    description:
      "Get opening hours for the library, gym, cafeteria, and every campus service — including weekends and holidays.",
    iconColor: "text-sky-600",
    iconBg: "bg-sky-50",
    border: "border-sky-100",
  },
  {
    icon: CalendarDays,
    title: "Discover Events",
    description:
      "Stay informed about upcoming lectures, social events, fairs, and student union activities happening on campus.",
    iconColor: "text-emerald-600",
    iconBg: "bg-emerald-50",
    border: "border-emerald-100",
  },
  {
    icon: Building2,
    title: "Explore Departments",
    description:
      "Find your faculty or department, get contact details, office hours, and navigate to the right building.",
    iconColor: "text-amber-600",
    iconBg: "bg-amber-50",
    border: "border-amber-100",
  },
  {
    icon: BookOpen,
    title: "Study Spaces",
    description:
      "Find the perfect place to study — from 24-hour suites to silent reading rooms and collaborative group pods.",
    iconColor: "text-rose-600",
    iconBg: "bg-rose-50",
    border: "border-rose-100",
  },
  {
    icon: Mic,
    title: "Voice & Image Input",
    description:
      "Ask your question by voice or upload a photo of a campus sign. The assistant understands all three modalities.",
    iconColor: "text-violet-600",
    iconBg: "bg-violet-50",
    border: "border-violet-100",
  },
];

export function Features() {
  return (
    <section id="features" className="bg-slate-50 py-24 px-6">
      <div className="mx-auto max-w-6xl">
        {/* Section header */}
        <div className="mb-16 text-center">
          <p className="mb-3 text-sm font-semibold uppercase tracking-widest text-indigo-600">
            Capabilities
          </p>
          <h2 className="text-4xl font-bold text-slate-900">
            Everything You Need to{" "}
            <span className="text-indigo-600">Navigate Campus</span>
          </h2>
          <p className="mt-4 text-slate-500 max-w-lg mx-auto">
            One assistant for every campus question — no more searching notice boards or asking strangers.
          </p>
        </div>

        {/* Feature grid */}
        <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
          {features.map(({ icon: Icon, title, description, iconColor, iconBg, border }) => (
            <div
              key={title}
              className={`rounded-2xl border ${border} bg-white p-6 shadow-sm hover:shadow-md transition-shadow`}
            >
              {/* Icon */}
              <div className={`mb-4 inline-flex rounded-xl ${iconBg} p-3`}>
                <Icon className={`h-5 w-5 ${iconColor}`} />
              </div>

              <h3 className="mb-2 font-semibold text-slate-900">{title}</h3>
              <p className="text-sm text-slate-500 leading-relaxed">{description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
