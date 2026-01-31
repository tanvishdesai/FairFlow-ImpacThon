"use client";

import { useState, useEffect } from "react";
import { 
  User, 
  CheckCircle,
  XCircle,
  AlertTriangle,
  Sparkles,
  Scale,
  TrendingUp,
  RefreshCw,
  Edit3,
  Play,
  Briefcase, 
  GraduationCap, 
  Award,
  Clock 
} from "lucide-react";
import Link from "next/link";
import { useForm } from "react-hook-form";

const API_BASE = "http://localhost:8000";

// Types for precomputed results
interface DemoCase {
  type: string; // "bias_victim", "correct_denial", "male_baseline", "custom"
  display_name: string;
  description: string;
  row_index: number | null; // null for custom cases
  gender: string;
  true_label: number | null; // null for custom cases unless simulated
  true_label_text: string;
  base_prediction: number;
  base_prediction_text: string;
  base_probability: number;
  fairflow_decision: number;
  fairflow_decision_text: string;
  intervention_occurred: boolean;
  intervention_type: string;
  current_dpr_at_decision: number;
  is_model_correct: boolean | null; // null for custom
  result_type: string;
  features?: {
    // Handling generic features map for flexibility, but focusing on recruitment keys
    // Job_Role_Applied, Experience_Years, Education_Level, Skill_Score, Interview_Score, etc.
    [key: string]: any; 
  };
}

interface ModelStats {
  accuracy: number;
  male_approval_rate: number;
  female_approval_rate: number;
  dpr: number;
}

interface FairFlowStats extends ModelStats {
  final_dpr: number;
  total_interventions: number;
}

interface PrecomputedData {
  base_model_stats: ModelStats;
  fairflow_stats: FairFlowStats;
  demo_cases: DemoCase[];
}

interface HiringFormData {
  age: number;
  gender: string; // 'Male', 'Female'
  education_level: string; // 'Bachelor', 'Master', 'PhD', 'High School' = mapped to int 0-3
  experience_years: number;
  skill_score: number;
  interview_score: number;
  job_role: string;
  expected_salary: number;
}

export default function SimulatorPage() {
  const [data, setData] = useState<PrecomputedData | null>(null);
  const [selectedCase, setSelectedCase] = useState<DemoCase | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // Tab state
  const [activeTab, setActiveTab] = useState<"demo" | "form">("demo");
  
  // Form state - updated for Hiring
  const { register, handleSubmit, formState: { errors, isSubmitting } } = useForm<HiringFormData>({
    defaultValues: {
      age: 28,
      gender: "Female",
      education_level: "Masters", // Changed from "2"
      experience_years: 5,
      skill_score: 85,
      interview_score: 80,
      job_role: "Software Engineer",
      expected_salary: 120000
    }
  });

  // Fetch precomputed demo cases on mount
  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/precomputed-demo-cases`); // This might need backend update to serve recruitment cases or we mock it for now if backend sends old ones
        // NOTE: If backend "precomputed-demo-cases" still returns Credit data, we should gracefully handle it.
        // Ideally backend should switch default demo cases based on active dataset (which we set to recruitment).
        
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const result = await res.json();
        setData(result);
        // Auto-select first bias victim case
        if (result.demo_cases?.length > 0) {
          const biasVictim = result.demo_cases.find((c: DemoCase) => c.type === "bias_victim");
          setSelectedCase(biasVictim || result.demo_cases[0]);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load demo cases");
      } finally {
        setIsLoading(false);
      }
    };
    fetchData();
  }, []);

  const onSubmit = async (formData: HiringFormData) => {
    try {
      // Map form values to features dictionary expected by backend prediction
      const payload = {
        features: {
            "Age": formData.age,
            "Gender": formData.gender, // Send as string "Male"/"Female"/"Other"
            "Education_Level": formData.education_level, // Send as string "Masters"/"Bachelors" etc
            "Experience_Years": formData.experience_years,
            "Skill_Score": formData.skill_score,
            "Interview_Score": formData.interview_score,
            "Job_Role_Applied": formData.job_role,
            "Expected_Salary": formData.expected_salary,
             // Defaults / Hidden fields for full feature set if needed
            "Technical_Test_Score": formData.skill_score, // Proxy
            "Aptitude_Test_Score": formData.skill_score, // Proxy
            "Communication_Score": formData.interview_score, // Proxy
            "Certifications_Count": 2, // Default
            "Previous_Companies": 1, // Default
            "Location": "Urban" // Default: "Urban", "Rural", "Semi-Urban"
        }
      };

      const res = await fetch(`${API_BASE}/api/predict`, { // Using /api/predict for custom single case
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const result = await res.json();
      
      // Adapt response to DemoCase format
      const customCase: DemoCase = {
        type: "custom",
        display_name: "Your Custom Candidate",
        description: "Live result from Fair Recruitment Model",
        row_index: null,
        gender: formData.gender,
        true_label: null,
        true_label_text: "UNKNOWN",
        base_prediction: result.base_prediction,
        base_prediction_text: result.base_prediction === 1 ? "HIRED" : "REJECTED",
        base_probability: result.base_probability,
        fairflow_decision: result.fairflow_decision,
        fairflow_decision_text: result.fairflow_decision === 1 ? "HIRED" : "REJECTED",
        intervention_occurred: result.intervened,
        intervention_type: result.intervention_type || "None",
        current_dpr_at_decision: 0.8, // Placeholder or fetch from metrics
        is_model_correct: null,
        result_type: "CUSTOM",
        features: {
            ...payload.features,
            // Add readable labels for display
            "Education_Level_Label": formData.education_level
        }
      };
      
      setSelectedCase(customCase);
      
    } catch (err) {
      console.error(err);
      alert("Simulation failed. See console.");
    }
  };

  const eduLabel = (level: number) => ["High School", "Bachelor's", "Master's", "PhD"][level] || "Unknown";

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center" style={{ background: 'var(--bg-primary)' }}>
        <div className="text-center">
          <RefreshCw className="w-12 h-12 animate-spin mx-auto mb-4" style={{ color: 'var(--primary)' }} />
          <p style={{ color: 'var(--text-secondary)' }}>Loading simulation environment...</p>
        </div>
      </div>
    );
  }

  // Fallback for data missing (e.g. error loading precomputed)
  if (!data) return <div className="p-8 text-center text-red-500">Failed to load data. Ensure backend is running.</div>;

  return (
    <div className="min-h-screen p-6" style={{ background: 'var(--bg-primary)' }}>
      <div className="max-w-[1400px] mx-auto">
        {/* Header */}
        <header className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <Link href="/" className="text-sm hover:underline" style={{ color: 'var(--text-secondary)' }}>
                ← Back to Dashboard
              </Link>
              <h1 className="text-3xl font-bold mt-2" style={{ color: 'var(--text-primary)' }}>
                <Briefcase className="inline-block mr-3 w-8 h-8" style={{ color: 'var(--primary)' }} />
                Recruitment Simulator
              </h1>
              <p className="mt-2" style={{ color: 'var(--text-secondary)' }}>
                Fair Recruitment AI: Simulating hiring decisions and mitigating gender bias
              </p>
            </div>
          </div>
        </header>

        {/* Stats Overview (Using data from backend - likely still Credit stats if precompute wasn't updated, but labels are generic enough) */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          <div className="glass-card p-4 text-center">
            <div className="text-sm mb-1" style={{ color: 'var(--text-secondary)' }}>Base Model DPR</div>
            <div className="text-3xl font-bold" style={{ color: 'var(--danger)' }}>
              {data.base_model_stats.dpr.toFixed(2)}
            </div>
            <div className="text-xs" style={{ color: 'var(--text-muted)' }}>Fairness Indicator</div>
          </div>
          <div className="glass-card p-4 text-center">
            <div className="text-sm mb-1" style={{ color: 'var(--text-secondary)' }}>FairFlow DPR</div>
            <div className="text-3xl font-bold" style={{ color: 'var(--success)' }}>
              {data.fairflow_stats.final_dpr.toFixed(2)}
            </div>
            <div className="text-xs" style={{ color: 'var(--text-muted)' }}>Fair &amp; Balanced</div>
          </div>
          <div className="glass-card p-4 text-center">
            <div className="text-sm mb-1" style={{ color: 'var(--text-secondary)' }}>Total Interventions</div>
            <div className="text-3xl font-bold" style={{ color: 'var(--primary)' }}>
              {data.fairflow_stats.total_interventions}
            </div>
            <div className="text-xs" style={{ color: 'var(--text-muted)' }}>Bias corrections applied</div>
          </div>
           {/* Hiring Rate Gap */}
          <div className="glass-card p-4 text-center">
            <div className="text-xs mb-1" style={{ color: 'var(--text-secondary)' }}>Hiring Gap (M vs F)</div>
            <div className="text-xl font-bold" style={{ color: 'var(--warning)' }}>
              {(data.base_model_stats.male_approval_rate * 100).toFixed(0)}% vs {(data.base_model_stats.female_approval_rate * 100).toFixed(0)}%
            </div>
            <div className="text-xs" style={{ color: 'var(--text-muted)' }}>Base Model Gap</div>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex gap-4 mb-6" style={{ borderBottom: '1px solid var(--border-subtle)' }}>
            <button
                onClick={() => setActiveTab("demo")}
                className={`pb-3 px-4 text-sm font-medium transition-colors relative ${
                    activeTab === "demo" ? "font-bold" : ""
                }`}
                style={{
                  color: activeTab === "demo" ? 'var(--primary)' : 'var(--text-secondary)',
                  opacity: activeTab === "demo" ? 1 : 0.7
                }}
            >
                <Sparkles className="inline-block w-4 h-4 mr-2" />
                Demo Cases
                {activeTab === "demo" && (
                    <div className="absolute bottom-0 left-0 right-0 h-0.5 rounded-t-full" style={{ backgroundColor: 'var(--primary)' }} />
                )}
            </button>
            <button
                onClick={() => setActiveTab("form")}
                className={`pb-3 px-4 text-sm font-medium transition-colors relative ${
                    activeTab === "form" ? "font-bold" : ""
                }`}
                style={{
                  color: activeTab === "form" ? 'var(--primary)' : 'var(--text-secondary)',
                  opacity: activeTab === "form" ? 1 : 0.7
                }}
            >
                <Edit3 className="inline-block w-4 h-4 mr-2" />
                Custom Candidate
                {activeTab === "form" && (
                    <div className="absolute bottom-0 left-0 right-0 h-0.5 rounded-t-full" style={{ backgroundColor: 'var(--primary)' }} />
                )}
            </button>
        </div>

        {/* Tab Content */}
        {activeTab === "demo" ? (
             /* Demo Cases Selector */
            <div className="glass-card p-4 mb-6">
                 <p className="text-sm text-gray-500 mb-4">
                     Select a pre-configured candidate profile to see how the model evaluates them.
                     {/* Note: If demo cases are still German Credit, this might look weird, but we are primarily testing 'Custom' flow for new dataset */}
                 </p>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                    {data.demo_cases.map((demoCase, idx) => (
                    <button
                        key={idx}
                        onClick={() => setSelectedCase(demoCase)}
                        className={`p-4 rounded-lg text-left transition-all`}
                        style={{ 
                          backgroundColor: selectedCase === demoCase 
                              ? 'rgba(99,102,241,0.15)' 
                              : 'var(--bg-secondary)',
                          borderLeft: `4px solid ${demoCase.type === 'bias_victim' ? 'var(--danger)' : demoCase.type === 'correct_denial' ? 'var(--success)' : 'var(--primary)'}`,
                          border: selectedCase === demoCase 
                              ? '1px solid var(--primary)' 
                              : '1px solid rgba(0,0,0,0)'
                        }}
                    >
                        <div className="flex items-center gap-2 mb-1">
                        <span className="font-medium" style={{ color: 'var(--text-primary)' }}>
                            {demoCase.display_name}
                        </span>
                        </div>
                        <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                        {demoCase.description}
                        </p>
                    </button>
                    ))}
                </div>
            </div>
        ) : (
            /* Interactive Form - RECRUITMENT SPECIFIC */
            <div className="glass-card p-6 mb-6">
                <div className="flex justify-between items-center mb-6">
                    <h3 className="text-lg font-semibold flex items-center gap-2" style={{ color: 'var(--text-primary)' }}>
                        <TrendingUp className="w-5 h-5" style={{ color: 'var(--primary)' }} />
                        Enter Candidate Details
                    </h3>
                </div>

                <form onSubmit={handleSubmit(onSubmit)} className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {/* Age */}
                    <div>
                        <label className="block text-xs mb-1" style={{ color: 'var(--text-secondary)' }}>Age</label>
                        <input 
                            type="number" 
                            {...register("age", { required: true, min: 18, max: 70 })}
                            className="w-full rounded px-3 py-2 text-sm outline-none transition-colors focus:border-indigo-500"
                            style={{ backgroundColor: 'var(--bg-glass)', color: 'var(--text-primary)', border: '1px solid var(--border-subtle)' }}
                        />
                    </div>

                     {/* Gender */}
                     <div>
                        <label className="block text-xs mb-1" style={{ color: 'var(--text-secondary)' }}>Gender</label>
                        <select 
                            {...register("gender")}
                            className="w-full rounded px-3 py-2 text-sm outline-none transition-colors focus:border-indigo-500"
                            style={{ backgroundColor: 'var(--bg-glass)', color: 'var(--text-primary)', border: '1px solid var(--border-subtle)' }}
                        >
                            <option value="Female" style={{ color: 'black' }}>Female</option>
                            <option value="Male" style={{ color: 'black' }}>Male</option>
                            <option value="Other" style={{ color: 'black' }}>Other</option>
                        </select>
                    </div>

                      {/* Job Role */}
                      <div>
                        <label className="block text-xs mb-1" style={{ color: 'var(--text-secondary)' }}>Job Role</label>
                        <input 
                            type="text" 
                            {...register("job_role")}
                            defaultValue="Software Engineer"
                            className="w-full rounded px-3 py-2 text-sm outline-none transition-colors focus:border-indigo-500"
                            style={{ backgroundColor: 'var(--bg-glass)', color: 'var(--text-primary)', border: '1px solid var(--border-subtle)' }}
                        />
                    </div>

                    {/* Education */}
                     <div>
                        <label className="block text-xs mb-1" style={{ color: 'var(--text-secondary)' }}>Education Level</label>
                        <select 
                            {...register("education_level")}
                            className="w-full rounded px-3 py-2 text-sm outline-none transition-colors focus:border-indigo-500"
                            style={{ backgroundColor: 'var(--bg-glass)', color: 'var(--text-primary)', border: '1px solid var(--border-subtle)' }}
                        >
                            <option value="PhD" style={{ color: 'black' }}>PhD</option>
                            <option value="Masters" style={{ color: 'black' }}>Masters</option>
                            <option value="Bachelors" style={{ color: 'black' }}>Bachelors</option>
                            <option value="High School" style={{ color: 'black' }}>High School</option>
                            <option value="Diploma" style={{ color: 'black' }}>Diploma</option>
                        </select>
                    </div>

                    {/* Experience */}
                    <div>
                        <label className="block text-xs mb-1" style={{ color: 'var(--text-secondary)' }}>Experience (Years)</label>
                        <input 
                            type="number" 
                            {...register("experience_years", { required: true, min: 0, max: 40 })}
                            className="w-full rounded px-3 py-2 text-sm outline-none transition-colors focus:border-indigo-500"
                            style={{ backgroundColor: 'var(--bg-glass)', color: 'var(--text-primary)', border: '1px solid var(--border-subtle)' }}
                        />
                    </div>

                    {/* Skill Score */}
                    <div>
                        <label className="block text-xs mb-1" style={{ color: 'var(--text-secondary)' }}>Skill Score (0-100)</label>
                        <input 
                            type="number" 
                            {...register("skill_score", { required: true, min: 0, max: 100 })}
                            className="w-full rounded px-3 py-2 text-sm outline-none transition-colors focus:border-indigo-500"
                            style={{ backgroundColor: 'var(--bg-glass)', color: 'var(--text-primary)', border: '1px solid var(--border-subtle)' }}
                        />
                    </div>

                    {/* Interview Score */}
                    <div>
                        <label className="block text-xs mb-1" style={{ color: 'var(--text-secondary)' }}>Interview Score (0-100)</label>
                        <input 
                            type="number" 
                            {...register("interview_score", { required: true, min: 0, max: 100 })}
                            className="w-full rounded px-3 py-2 text-sm outline-none transition-colors focus:border-indigo-500"
                            style={{ backgroundColor: 'var(--bg-glass)', color: 'var(--text-primary)', border: '1px solid var(--border-subtle)' }}
                        />
                    </div>
                
                    <div className="md:col-span-3 flex justify-end mt-4">
                        <button 
                            type="submit"
                            disabled={isSubmitting}
                            className="px-6 py-2 rounded-lg font-medium transition-all flex items-center gap-2 shadow-lg"
                            style={{
                              background: 'var(--primary)',
                              color: '#fff',
                              boxShadow: '0 4px 14px rgba(99, 102, 241, 0.3)'
                            }}
                        >
                            {isSubmitting ? (
                                <RefreshCw className="w-5 h-5 animate-spin" />
                            ) : (
                                <Play className="w-5 h-5 fill-current" />
                            )}
                            Evaluate Candidate
                        </button>
                    </div>
                </form>
            </div>
        )}

        {/* Selected Case Details */}
        {selectedCase && (
          <div className="space-y-6 animate-slide-up">
            {/* Applicant Info */}
            <div className="glass-card p-4">
              <div className="flex items-center gap-4">
                <User className="w-8 h-8" style={{ color: selectedCase.gender === 'Female' ? '#ec4899' : 'var(--primary)' }} />
                <div>
                  <h3 className="text-lg font-semibold" style={{ color: 'var(--text-primary)' }}>
                    {selectedCase.display_name}
                  </h3>
                  <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
                    {selectedCase.gender} Candidate
                    {selectedCase.result_type === 'CUSTOM' && ' (Simulated)'}
                  </p>
                </div>
              </div>
            </div>

            {/* Feature Values */}
            {selectedCase.features && (
              <div className="glass-card p-4">
                <h4 className="font-semibold mb-3 flex items-center gap-2" style={{ color: 'var(--text-primary)' }}>
                  <TrendingUp className="w-4 h-4" style={{ color: 'var(--primary)' }} />
                  Candidate Profile
                </h4>
                
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {/* Render known fields nicely */}
                  <div className="p-3 rounded-lg" style={{ backgroundColor: 'var(--bg-secondary)' }}>
                    <div className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>Role</div>
                    <div className="text-sm font-bold" style={{ color: 'var(--text-primary)' }}>
                      {selectedCase.features.Job_Role_Applied || selectedCase.features.Job || "N/A"}
                    </div>
                  </div>
                   <div className="p-3 rounded-lg" style={{ backgroundColor: 'var(--bg-secondary)' }}>
                    <div className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>Experience</div>
                    <div className="text-lg font-bold" style={{ color: 'var(--primary)' }}>
                      {selectedCase.features.Experience_Years ? `${selectedCase.features.Experience_Years} yrs` : "N/A"}
                    </div>
                  </div>
                   <div className="p-3 rounded-lg" style={{ backgroundColor: 'var(--bg-secondary)' }}>
                    <div className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>Education</div>
                    <div className="text-sm font-bold" style={{ color: 'var(--text-primary)' }}>
                      {selectedCase.features.Education_Level_Label 
                        ? selectedCase.features.Education_Level_Label 
                        : (selectedCase.features.Education_Level !== undefined ? eduLabel(selectedCase.features.Education_Level) : "N/A")}
                    </div>
                  </div>
                   <div className="p-3 rounded-lg" style={{ backgroundColor: 'var(--bg-secondary)' }}>
                    <div className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>Skill Score</div>
                    <div className="text-lg font-bold" style={{ color: selectedCase.features.Skill_Score > 80 ? 'var(--success)' : 'var(--warning)' }}>
                      {selectedCase.features.Skill_Score || "N/A"}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Three-Panel Decision Flow (Simplified for Hiring) */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              
              {/* Base Model Panel */}
              <div 
                className="glass-card p-6 relative overflow-hidden"
                style={{ 
                  borderColor: selectedCase.base_prediction === 1 ? 'var(--success)' : 'var(--danger)',
                  borderWidth: '2px'
                }}
              >
                <div className="absolute top-0 left-0 right-0 h-1" style={{ 
                  backgroundColor: selectedCase.base_prediction === 1 ? 'var(--success)' : 'var(--danger)' 
                }} />
                <h3 className="text-sm font-medium mb-4 flex items-center gap-2" style={{ color: 'var(--text-secondary)' }}>
                  BASE MODEL DECISION
                </h3>
                <div className="flex items-center gap-3 mb-3">
                  {selectedCase.base_prediction === 1 ? (
                    <CheckCircle className="w-12 h-12" style={{ color: 'var(--success)' }} />
                  ) : (
                    <XCircle className="w-12 h-12" style={{ color: 'var(--danger)' }} />
                  )}
                  <div>
                    <div className="text-2xl font-bold" style={{ 
                      color: selectedCase.base_prediction === 1 ? 'var(--success)' : 'var(--danger)' 
                    }}>
                      {selectedCase.base_prediction_text || (selectedCase.base_prediction === 1 ? "HIRED" : "REJECTED")}
                    </div>
                    <div className="text-sm" style={{ color: 'var(--text-secondary)' }}>
                      Initial Assessment
                    </div>
                  </div>
                </div>
              </div>

              {/* FairFlow Panel */}
              <div 
                className={`glass-card p-6 relative overflow-hidden ${selectedCase.intervention_occurred ? 'animate-pulse-once' : ''}`}
                style={{ 
                  borderColor: selectedCase.intervention_occurred ? 'var(--primary)' : (selectedCase.fairflow_decision === 1 ? 'var(--success)' : 'var(--danger)'),
                  borderWidth: '2px',
                  boxShadow: selectedCase.intervention_occurred ? '0 0 20px rgba(99,102,241,0.3)' : undefined
                }}
              >
                <div className="absolute top-0 left-0 right-0 h-1" style={{ 
                  backgroundColor: selectedCase.intervention_occurred ? 'var(--primary)' : (selectedCase.fairflow_decision === 1 ? 'var(--success)' : 'var(--danger)')
                }} />
                <h3 className="text-sm font-medium mb-4 flex items-center gap-2" style={{ color: 'var(--text-secondary)' }}>
                  FAIRFLOW FINAL DECISION
                  {selectedCase.intervention_occurred && (
                    <span className="px-2 py-0.5 text-xs rounded-full" style={{ backgroundColor: 'rgba(99,102,241,0.2)', color: 'var(--primary)' }}>
                      OVERRIDE
                    </span>
                  )}
                </h3>
                <div className="flex items-center gap-3 mb-3">
                  {selectedCase.fairflow_decision === 1 ? (
                    <CheckCircle className="w-12 h-12" style={{ color: 'var(--success)' }} />
                  ) : (
                    <XCircle className="w-12 h-12" style={{ color: 'var(--danger)' }} />
                  )}
                  <div>
                    <div className="text-2xl font-bold" style={{ 
                      color: selectedCase.fairflow_decision === 1 ? 'var(--success)' : 'var(--danger)' 
                    }}>
                      {selectedCase.fairflow_decision_text || (selectedCase.fairflow_decision === 1 ? "HIRED" : "REJECTED")}
                    </div>
                    <div className="text-sm" style={{ color: 'var(--text-secondary)' }}>
                      {selectedCase.intervention_type}
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Intervention Explanation */}
            {selectedCase.intervention_occurred && (
              <div 
                className="glass-card p-5 border-l-4"
                style={{ borderLeftColor: 'var(--primary)', backgroundColor: 'rgba(99,102,241,0.05)' }}
              >
                <div className="flex items-start gap-3">
                  <AlertTriangle className="w-6 h-6 flex-shrink-0 mt-0.5" style={{ color: 'var(--warning)' }} />
                  <div>
                    <h4 className="font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>
                      Fairness Intervention Applied
                    </h4>
                    <p className="text-sm leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
                      The base model rejected this candidate, but FairFlow detected that the rejection was likely influenced by gender bias rather than qualification gaps. The decision was overridden to ensure equal opportunity.
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
